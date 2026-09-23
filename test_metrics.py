import numpy as np
import pandas as pd
from metrics import weekly_prices, calculate, state


def fixture():
    idx=pd.bdate_range('2023-01-01','2026-10-01')
    return pd.DataFrame({'QQQ':100.,'XLK':np.linspace(100,200,len(idx))},index=idx)


def test_week_and_provisional():
    d=fixture()
    w,t,p=weekly_prices(d,'2026-09-22 22:00Z')
    assert t==pd.Timestamp('2026-09-18') and not p
    w,t,p=weekly_prices(d,'2026-09-22 22:00Z',True)
    assert t==pd.Timestamp('2026-09-22') and p


def test_good_friday():
    w,t,p=weekly_prices(fixture(),'2026-04-03 22:00Z')
    assert t==pd.Timestamp('2026-04-02') and not p


def test_missing_close_not_filled():
    d=fixture();d.loc['2026-09-18','XLK']=np.nan
    w,_,_=weekly_prices(d,'2026-09-22 22:00Z')
    assert pd.isna(w.XLK.iloc[-1])
    assert pd.isna(calculate(w)[1].XLK.iloc[-1])


def test_same_performance_and_outperformance():
    idx=pd.date_range('2023-01-06',periods=60,freq='W-FRI')
    w=pd.DataFrame({'QQQ':100*1.01**np.arange(60),'SAME':50*1.01**np.arange(60),'FAST':100*1.02**np.arange(60)},index=idx)
    periods,s,d=calculate(w)
    assert abs(s.SAME.iloc[-1]-1)<1e-10
    assert s.FAST.iloc[-1]>1
    assert state(1.1,-.4)=='強いが悪化中'


def test_app():
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    from universe import SECTOR_ETFS,THEME_ETFS
    daily=fixture()
    for i,s in enumerate({**SECTOR_ETFS,**THEME_ETFS}): daily[s]=100*np.exp(np.arange(len(daily))*(i-20)/100000)
    raw=pd.concat({'Close':daily},axis=1)
    with patch('yfinance.download',return_value=raw):
        app=AppTest.from_file('app.py',default_timeout=30).run()
        assert not app.exception
        app.multiselect[0].set_value([]).run()
        assert not app.exception
        app.toggle[0].set_value(True).run()
        assert not app.exception
