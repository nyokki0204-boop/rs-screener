import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warnings
import os
import datetime
warnings.filterwarnings('ignore')

try:
    import japanize_matplotlib
except:
    pass

st.set_page_config(page_title="RS Screener", page_icon="📈", layout="wide")
st.title("📈 RS Screener")
st.caption("セクター・テーマETFの相対強度（vs QQQ）")

BENCHMARK     = 'QQQ'
RS_PERIODS    = [4, 13, 26, 52]
TOP_N         = 5
HISTORY_WEEKS = 26
HISTORY_PATH  = 'data/rs_history.csv'

SECTOR_ETFS = {
    'XLK':'テクノロジー','XLV':'ヘルスケア','XLF':'金融',
    'XLY':'一般消費財','XLC':'コミュニケーション','XLI':'資本財',
    'XLP':'生活必需品','XLE':'エネルギー','XLU':'公益事業',
    'XLRE':'不動産','XLB':'素材',
}

THEME_ETFS = {
    'SMH':'半導体','SOXX':'半導体(iShares)','ARKK':'ARKイノベーション',
    'ARKG':'ARKゲノム','ARKW':'ARK次世代インターネット','BOTZ':'AI・ロボット',
    'AIQ':'AI全般','ROBO':'ロボティクス','WCLD':'クラウド',
    'CLOU':'クラウドコンピューティング','CIBR':'サイバーセキュリティ',
    'HACK':'サイバーセキュリティ2','FINX':'フィンテック','IPAY':'デジタル決済',
    'ICLN':'クリーンエネルギー','QCLN':'クリーンエネルギー2','LIT':'リチウム・EV',
    'DRIV':'自動運転・EV','KARS':'EV全般','IBB':'バイオテク','XBI':'バイオテク2',
    'GLD':'金(ゴールド)','SLV':'銀','GDX':'金鉱株','USO':'原油','DBA':'農産物',
    'JETS':'航空','ITB':'住宅建設','XHB':'ホームビルダー','XRT':'小売',
    'HERO':'ゲーム・eスポーツ','ESPO':'ゲーム2','METV':'メタバース',
    'UFO':'宇宙','BETZ':'スポーツ賭博','MJ':'大麻',
}

ALL_ETFS = {**SECTOR_ETFS, **THEME_ETFS}

@st.cache_data(ttl=3600)
def load_data():
    bm_raw = yf.download(BENCHMARK, period='3y', interval='1wk',
                         progress=False, auto_adjust=True)
    if isinstance(bm_raw.columns, pd.MultiIndex):
        bm_raw.columns = bm_raw.columns.get_level_values(0)
    bm_close = bm_raw['Close'].astype(float).dropna()
    raw_all = yf.download(list(ALL_ETFS.keys()), period='3y', interval='1wk',
                          progress=False, auto_adjust=True)
    if isinstance(raw_all.columns, pd.MultiIndex):
        close_all = raw_all['Close'].astype(float)
    else:
        close_all = raw_all[['Close']].astype(float)
    return bm_close, close_all

def calc_rs_row(sym, name, category, bm_close, close_all):
    try:
        if sym not in close_all.columns:
            return None
        ec     = close_all[sym].dropna()
        bc     = bm_close.reindex(ec.index, method='ffill').dropna()
        common = ec.index.intersection(bc.index)
        if len(common) < 53:
            return None
        ec = ec[common]
        bc = bc[common]
        row = {'シンボル': sym, '名称': name, 'カテゴリ': category}
        for p in RS_PERIODS:
            if len(common) >= p + 1:
                rs = (ec.iloc[-1] / ec.iloc[-p-1]) / (bc.iloc[-1] / bc.iloc[-p-1])
                row[f'RS_{p}週'] = round(float(rs), 3)
            else:
                row[f'RS_{p}週'] = None
        rs_vals = [row[f'RS_{p}週'] for p in RS_PERIODS if row[f'RS_{p}週'] is not None]
        row['総合RS'] = round(sum(rs_vals) / len(rs_vals), 3) if rs_vals else None
        row['現在値'] = round(float(ec.iloc[-1]), 2)
        row['52週高値比%'] = round((ec.iloc[-1] - ec.iloc[-52:].max()) / ec.iloc[-52:].max() * 100, 1)
        return row
    except:
        return None

def calc_rs_series(sym, bm_close, close_all, weeks=HISTORY_WEEKS):
    try:
        if sym not in close_all.columns:
            return None
        ec     = close_all[sym].dropna()
        bc     = bm_close.reindex(ec.index, method='ffill').dropna()
        common = ec.index.intersection(bc.index)
        if len(common) < weeks + 5:
            return None
        ec = ec[common].iloc[-weeks-1:]
        bc = bc[common].iloc[-weeks-1:]
        rs = (ec / ec.iloc[0]) / (bc / bc.iloc[0])
        return rs.iloc[1:]
    except:
        return
