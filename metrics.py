"""Calendar-aligned weekly relative strength. No forward filling of prices."""
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

PERIODS = (4, 13, 26, 52)
NEUTRAL = 0.2  # percentage-point change in composite RS


def weekly_prices(daily, now=None, provisional=False):
    now = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz='UTC'))
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    data = daily.copy()
    data.index = pd.DatetimeIndex(data.index).tz_localize(None).normalize()
    data = data[~data.index.duplicated(keep='last')].sort_index()
    data = data.where(np.isfinite(data) & (data > 0))
    if data.empty:
        raise ValueError('価格データがありません。')
    schedule = mcal.get_calendar('NYSE').schedule(start_date=data.index.min(), end_date=now.tz_convert('America/New_York').date())
    closed = schedule[schedule.market_close <= now]
    if closed.empty:
        raise ValueError('確定した取引日のデータがありません。')
    last_day = closed.index[-1]
    week_end = last_day.to_period('W-FRI').end_time.normalize()
    full = mcal.get_calendar('NYSE').schedule(start_date=last_day - pd.Timedelta(days=7), end_date=week_end)
    final_day = full.loc[full.index.to_period('W-FRI') == last_day.to_period('W-FRI')].index[-1]
    is_partial = last_day < final_day
    target = last_day if provisional or not is_partial else closed.loc[closed.index.to_period('W-FRI') < last_day.to_period('W-FRI')].index[-1]
    sessions = closed.loc[:target]
    ends = sessions.groupby(sessions.index.to_period('W-FRI')).tail(1).index
    # Reindex first: a missing weekly closing session must remain NaN.
    weekly = data.reindex(ends)
    weekly.index = ends.to_period('W-FRI').to_timestamp(how='end').normalize()
    weekly = weekly.reindex(pd.date_range(weekly.index.min(), weekly.index.max(), freq='W-FRI'))
    return weekly, target, bool(provisional and is_partial)


def calculate(weekly, benchmark='QQQ'):
    ratio = weekly.div(weekly[benchmark], axis=0).drop(columns=[benchmark])
    periods = {p: ratio.div(ratio.shift(p)) for p in PERIODS}
    # Require every calendar week in the full 53-observation window.
    valid = ratio.notna().rolling(53).sum().eq(53)
    score = sum(periods.values()) / len(PERIODS)
    score = score.where(valid)
    return periods, score, score.diff() * 100


def state(score, change):
    if pd.isna(score) or pd.isna(change):
        return '比較データ不足'
    if abs(change) <= NEUTRAL:
        return '強さを維持' if score >= 1 else '弱いまま横ばい'
    return ('強く、さらに改善' if score >= 1 else '弱いが回復中') if change > 0 else ('強いが悪化中' if score >= 1 else '弱く、さらに悪化')
