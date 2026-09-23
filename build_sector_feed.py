"""Build the small public, read-only sector feed consumed by MyChart."""

import io
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from metrics import PERIODS, NEUTRAL, calculate, state, weekly_prices
from universe import SECTOR_ETFS


HOLDINGS_URL = (
    'https://www.ssga.com/library-content/products/fund-data/etfs/us/'
    'holdings-daily-us-en-{ticker}.xlsx'
)
OUTPUT = Path(__file__).parent / 'data' / 'sector_feed.json'
HOLDINGS_OUTPUT = Path(__file__).parent / 'data' / 'sector_holdings.json'


def parse_holdings(payload, sector):
    """Only real, positively weighted tickers; never infer from security names."""
    rows = pd.read_excel(io.BytesIO(payload), header=None).fillna('').values.tolist()
    header = next((i for i, row in enumerate(rows)
                   if str(row[0]).strip() == 'Name' and str(row[1]).strip() == 'Ticker'), None)
    if header is None or header < 2:
        raise ValueError(f'{sector}: 保有銘柄の列がありません')
    weight_col = next((i for i, value in enumerate(rows[header]) if value == 'Weight'), None)
    if weight_col is None:
        raise ValueError(f'{sector}: Weight列がありません')
    date_text = str(rows[header - 2][1]).replace('As of ', '').strip()
    date = pd.to_datetime(date_text, errors='coerce')
    if pd.isna(date):
        raise ValueError(f'{sector}: 基準日を読み取れません')
    tickers = set()
    for row in rows[header + 1:]:
        ticker = str(row[1]).strip().upper()
        weight = pd.to_numeric(row[weight_col], errors='coerce')
        if re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}', ticker) and pd.notna(weight) and weight > 0:
            if ticker not in {'USD', 'CASH', 'US DOLLAR'}:
                tickers.add(ticker.replace('.', '-'))
    if len(tickers) < 10:
        raise ValueError(f'{sector}: 保有銘柄が少なすぎます ({len(tickers)})')
    return tickers, date.date().isoformat()


def fetch_holdings():
    assignments = {}
    dates = []
    for sector in SECTOR_ETFS:
        req = Request(HOLDINGS_URL.format(ticker=sector.lower()),
                      headers={'User-Agent': 'MyChart sector research (public holdings)'})
        with urlopen(req, timeout=30) as response:
            tickers, date = parse_holdings(response.read(), sector)
        dates.append(date)
        for ticker in tickers:
            if ticker in assignments and assignments[ticker] != sector:
                raise ValueError(f'{ticker}: 複数のセクターETFに含まれます')
            assignments[ticker] = sector
    if len(assignments) < 450 or (pd.Timestamp.now(tz='UTC').date() - pd.Timestamp(min(dates)).date()).days > 14:
        raise ValueError('保有銘柄の網羅性または鮮度が不足しています')
    return dict(sorted(assignments.items())), min(dates)


def build_feed(daily, holdings, holdings_date, now=None):
    weekly, price_date, partial = weekly_prices(daily, now=now, provisional=False)
    if partial:
        raise ValueError('確定週足ではありません')
    periods, scores, changes = calculate(weekly)
    latest = scores.iloc[-1].reindex(SECTOR_ETFS.keys()).dropna().sort_values(ascending=False)
    if len(latest) != len(SECTOR_ETFS) or pd.isna(weekly['QQQ'].iloc[-1]):
        raise ValueError('11セクターとQQQの確定週足が揃っていません')
    sectors = {}
    for rank, (ticker, score) in enumerate(latest.items(), 1):
        delta = changes[ticker].iloc[-1]
        if pd.isna(delta):
            raise ValueError(f'{ticker}: 前週差がありません')
        sectors[ticker] = {
            'name': SECTOR_ETFS[ticker], 'rank': rank, 'score': round(float(score), 6),
            'change_pt': round(float(delta), 4), 'state': state(score, delta),
            'periods': {str(period): round(float(periods[period][ticker].iloc[-1]), 6)
                        for period in PERIODS},
            'return_4w_pct': round(float((weekly[ticker].iloc[-1] / weekly[ticker].iloc[-5] - 1) * 100), 3),
        }
    return {
        'version': 1, 'benchmark': 'QQQ', 'price_date': price_date.date().isoformat(),
        'week_label': weekly.index[-1].date().isoformat(), 'holdings_date': holdings_date,
        'updated_at': datetime.now(timezone.utc).isoformat(), 'neutral_change_pt': NEUTRAL,
        'sectors': sectors, 'holdings': holdings,
    }


def write_json(path, data):
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdings-only', action='store_true')
    args = parser.parse_args()
    if args.holdings_only:
        holdings, date = fetch_holdings()
        write_json(HOLDINGS_OUTPUT, {'version': 1, 'holdings_date': date, 'holdings': holdings})
        print(f'{date}: {len(holdings)}銘柄のセクターを更新')
        return
    record = json.loads(HOLDINGS_OUTPUT.read_text(encoding='utf-8'))
    holdings, holdings_date = record['holdings'], record['holdings_date']
    if len(holdings) < 450 or (pd.Timestamp.now(tz='UTC').date() - pd.Timestamp(holdings_date).date()).days > 14:
        raise ValueError('保有銘柄データが古いか不足しています')
    raw = yf.download(list(SECTOR_ETFS) + ['QQQ'], period='3y', interval='1d',
                      auto_adjust=True, progress=False, threads=False)
    if raw.empty or 'Close' not in raw:
        raise ValueError('ETFの価格データを取得できませんでした')
    feed = build_feed(raw['Close'].reindex(columns=list(SECTOR_ETFS) + ['QQQ']),
                      holdings, holdings_date)
    write_json(OUTPUT, feed)
    print(f'{feed["price_date"]}: 11セクター、{len(holdings)}銘柄を更新')


if __name__ == '__main__':
    main()
