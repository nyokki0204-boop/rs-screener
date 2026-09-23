import io

import numpy as np
import pandas as pd
import pytest

from build_sector_feed import build_feed, parse_holdings
from universe import SECTOR_ETFS


def test_parse_holdings_requires_real_tickers_and_weight():
    rows = [['Fund Name:', 'XLK'], ['Ticker Symbol:', 'XLK'], ['Holdings:', 'As of 22-Sep-2026'],
            ['', ''], ['Name', 'Ticker', 'Weight']]
    rows += [[f'Company {i}', f'T{i}', 1] for i in range(11)]
    rows += [['US DOLLAR', 'USD', 1], ['Zero', 'FAKE', 0]]
    workbook = io.BytesIO()
    pd.DataFrame(rows).to_excel(workbook, header=False, index=False)
    tickers, date = parse_holdings(workbook.getvalue(), 'XLK')
    assert len(tickers) == 11
    assert 'USD' not in tickers and 'FAKE' not in tickers
    assert date == '2026-09-22'


def test_feed_matches_existing_weekly_calculation():
    dates = pd.date_range('2024-09-01', '2026-09-18', freq='B')
    daily = pd.DataFrame(index=dates)
    daily['QQQ'] = 100 * np.exp(np.arange(len(dates)) * .0004)
    for i, ticker in enumerate(SECTOR_ETFS):
        daily[ticker] = 100 * np.exp(np.arange(len(dates)) * (.0002 + i * .00002))
    feed = build_feed(daily, {'AAPL':'XLK'}, '2026-09-18', now='2026-09-19T12:00:00Z')
    assert len(feed['sectors']) == 11
    assert feed['benchmark'] == 'QQQ'
    assert sorted(x['rank'] for x in feed['sectors'].values()) == list(range(1,12))
    assert all(set(s['periods']) == {'4','13','26','52'} for s in feed['sectors'].values())


def test_feed_refuses_missing_sector_price():
    dates = pd.date_range('2024-09-01', '2026-09-18', freq='B')
    daily = pd.DataFrame(100., index=dates, columns=['QQQ', *SECTOR_ETFS])
    daily.loc[daily.index[-1], 'XLK'] = np.nan
    with pytest.raises(ValueError, match='11セクター'):
        build_feed(daily, {}, '2026-09-18', now='2026-09-19T12:00:00Z')
