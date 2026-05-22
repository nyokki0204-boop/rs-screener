import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import japanize_matplotlib
except:
    pass

st.set_page_config(
    page_title="RS Screener",
    page_icon="📈",
    layout="wide"
)

st.title("📈 RS Screener")
st.caption("セクター・テーマETFの相対強度（vs QQQ）")

# ============================================================
#  設定
# ============================================================
BENCHMARK     = 'QQQ'
RS_PERIODS    = [4, 13, 26, 52]
TOP_N         = 5
HISTORY_WEEKS = 26

SECTOR_ETFS = {
    'XLK'  : 'テクノロジー',
    'XLV'  : 'ヘルスケア',
    'XLF'  : '金融',
    'XLY'  : '一般消費財',
    'XLC'  : 'コミュニケーション',
    'XLI'  : '資本財',
    'XLP'  : '生活必需品',
    'XLE'  : 'エネルギー',
    'XLU'  : '公益事業',
    'XLRE' : '不動産',
    'XLB'  : '素材',
}

THEME_ETFS = {
    'SMH'  : '半導体',
    'SOXX' : '半導体(iShares)',
    'ARKK' : 'ARKイノベーション',
    'ARKG' : 'ARKゲノム',
    'ARKW' : 'ARK次世代インターネット',
    'BOTZ' : 'AI・ロボット',
    'AIQ'  : 'AI全般',
    'ROBO' : 'ロボティクス',
    'WCLD' : 'クラウド',
    'CLOU' : 'クラウドコンピューティング',
    'CIBR' : 'サイバーセキュリティ',
    'HACK' : 'サイバーセキュリティ2',
    'FINX' : 'フィンテック',
    'IPAY' : 'デジタル決済',
    'ICLN' : 'クリーンエネルギー',
    'QCLN' : 'クリーンエネルギー2',
    'LIT'  : 'リチウム・EV',
    'DRIV' : '自動運転・EV',
    'KARS' : 'EV全般',
    'IBB'  : 'バイオテク',
    'XBI'  : 'バイオテク2',
    'GLD'  : '金(ゴールド)',
    'SLV'  : '銀',
    'GDX'  : '金鉱株',
    'USO'  : '原油',
    'DBA'  : '農産物',
    'JETS' : '航空',
    'ITB'  : '住宅建設',
    'XHB'  : 'ホームビルダー',
    'XRT'  : '小売',
    'HERO' : 'ゲーム・eスポーツ',
    'ESPO' : 'ゲーム2',
    'METV' : 'メタバース',
    'UFO'  : '宇宙',
    'BETZ' : 'スポーツ賭博',
    'MJ'   : '大麻',
}

ALL_ETFS = {**SECTOR_ETFS, **THEME_ETFS}

# ============================================================
#  データ取得（キャッシュで高速化）
# ============================================================
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

# ============================================================
#  RS計算
# ============================================================
def calc_rs_row(sym, name, category, bm_close, close_all):
    try:
        if sym not in close_all.columns:
            return None
        ec     = close_all[sym].dropna()
        bc     = bm_close.reindex(ec.index, method='ffill').dropna()
        common = ec.index.intersection(bc.index)
        if len(common) < 53:
            return None
        ec  = ec[common]
        bc  = bc[common]
        row = {'シンボル': sym, '名称': name, 'カテゴリ': category}
        for p in RS_PERIODS:
            if len(common) >= p + 1:
                rs = (ec.iloc[-1] / ec.iloc[-p-1]) / (bc.iloc[-1] / bc.iloc[-p-1])
                row[f'RS_{p}週'] = round(float(rs), 3)
            else:
                row[f'RS_{p}週'] = None
        rs_vals      = [row[f'RS_{p}週'] for p in RS_PERIODS if row[f'RS_{p}週'] is not None]
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
        return None

# ============================================================
#  グラフ描画
# ============================================================
def draw_chart(etf_dict, category_label, df_cat, bm_close, close_all):
    top   = df_cat.nlargest(TOP_N, '総合RS')
    syms  = top['シンボル'].tolist()
    names = [etf_dict.get(s, s) for s in syms]
    palette = ['#00ff88','#ff6b6b','#4ecdc4','#ffd93d','#a29bfe']

    fig = plt.figure(figsize=(12, 9), facecolor='#0d1117')
    gs  = gridspec.GridSpec(2, 1, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in [ax1, ax2]:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        ax.grid(True, alpha=0.12, color='#444444')
        for spine in ax.spines.values():
            spine.set_color('#2a2a2a')

    ax1.set_title(f'{category_label} RS推移 上位{TOP_N}本（vs {BENCHMARK}）',
                  color='white', fontsize=11, fontweight='bold', pad=10)
    ax1.axhline(1.0, color='#555555', linewidth=1.0, linestyle='--')

    bar_data = []
    rs_last  = None
    for i, (sym, name, color) in enumerate(zip(syms, names, palette)):
        rs = calc_rs_series(sym, bm_close, close_all)
        if rs is None:
            continue
        rs_last = rs
        ax1.plot(rs.index, rs.values, color=color, linewidth=2.2, label=f'{sym} {name}')
        ax1.annotate(f'{rs.iloc[-1]:.2f}',
                     xy=(rs.index[-1], rs.iloc[-1]),
                     xytext=(6, 0), textcoords='offset points',
                     color=color, fontsize=9, fontweight='bold')
        ax1.plot(rs.index[-4:], rs.values[-4:], color=color, linewidth=4.0, alpha=0.5)
        bar_data.append((sym, name, color, rs.diff().iloc[-8:]))

    ax1.set_ylabel('RS値', color='#aaaaaa', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8, facecolor='#1a1a1a',
               labelcolor='white', framealpha=0.85, edgecolor='#333333')
    plt.setp(ax1.get_xticklabels(), rotation=30)

    ax2.set_title('勢い（週次RS変化）  ＋ = 加速  − = 減速',
                  color='#aaaaaa', fontsize=10, pad=8)
    ax2.axhline(0, color='#555555', linewidth=1.0)

    if bar_data:
        n_bars = len(bar_data[0][3])
        x      = np.arange(n_bars)
        width  = 0.15
        offset = -(len(bar_data)-1) / 2 * width
        for j, (sym, name, color, changes) in enumerate(bar_data):
            bars = ax2.bar(x + offset + j * width, changes.values,
                           width, color=color, alpha=0.8, label=sym)
            bars[-1].set_alpha(1.0)
            bars[-1].set_edgecolor('white')
            bars[-1].set_linewidth(1.0)
        if rs_last is not None:
            dates = [d.strftime('%m/%d') for d in bar_data[0][3].index]
            ax2.set_xticks(x)
            ax2.set_xticklabels(dates, rotation=30, color='#aaaaaa', fontsize=9)
        ax2.set_ylabel('RS週次変化', color='#aaaaaa', fontsize=10)
        ax2.legend(loc='upper left', fontsize=8, facecolor='#1a1a1a',
                   labelcolor='white', framealpha=0.85, edgecolor='#333333',
                   ncol=len(bar_data))
        ymin, ymax = ax2.get_ylim()
        ax2.axhspan(0, max(ymax, 0.01), alpha=0.05, color='#00ff88')
        ax2.axhspan(min(ymin, -0.01), 0, alpha=0.05, color='#ff6b6b')

    plt.tight_layout()
    return fig

# ============================================================
#  メイン画面
# ============================================================
if st.button('🔄 データを更新', type='primary', use_container_width=True):
    st.cache_data.clear()

with st.spinner('データ取得中... (約30秒)'):
    bm_close, close_all = load_data()

results = []
for sym, name in SECTOR_ETFS.items():
    r = calc_rs_row(sym, name, 'セクター', bm_close, close_all)
    if r: results.append(r)
for sym, name in THEME_ETFS.items():
    r = calc_rs_row(sym, name, 'テーマ', bm_close, close_all)
    if r: results.append(r)

df_rs     = pd.DataFrame(results)
df_sector = df_rs[df_rs['カテゴリ']=='セクター'].sort_values('総合RS', ascending=False).reset_index(drop=True)
df_theme  = df_rs[df_rs['カテゴリ']=='テーマ'].sort_values('総合RS', ascending=False).reset_index(drop=True)
df_sector.index += 1
df_theme.index  += 1

cols    = ['シンボル','名称','総合RS','RS_4週','RS_13週','RS_26週','RS_52週','52週高値比%']
rs_cols = ['総合RS','RS_4週','RS_13週','RS_26週','RS_52週']

def color_rs(val):
    if isinstance(val, float):
        if val >= 1.10: return 'background-color:#1a6e1a;color:white'
        elif val >= 1.0: return 'background-color:#4a9e4a;color:white'
        elif val >= 0.9: return 'background-color:#8B0000;color:white'
        else: return 'background-color:#5c0000;color:white'
    return ''

# タブで切り替え
tab1, tab2 = st.tabs(['📊 セクター', '🎯 テーマ'])

with tab1:
    st.subheader(f'📊 セクター RS ランキング（vs {BENCHMARK}）')
    st.dataframe(
        df_sector[cols].style
        .applymap(color_rs, subset=rs_cols)
        .format({c: '{:.3f}' for c in rs_cols} | {'52週高値比%': '{:.1f}%'}),
        use_container_width=True, height=420
    )
    fig1 = draw_chart(SECTOR_ETFS, '📊 セクター', df_sector, bm_close, close_all)
    st.pyplot(fig1)

with tab2:
    st.subheader(f'🎯 テーマ RS ランキング（vs {BENCHMARK}）')
    st.dataframe(
        df_theme[cols].style
        .applymap(color_rs, subset=rs_cols)
        .format({c: '{:.3f}' for c in rs_cols} | {'52週高値比%': '{:.1f}%'}),
        use_container_width=True, height=420
    )
    fig2 = draw_chart(THEME_ETFS, '🎯 テーマ', df_theme, bm_close, close_all)
    st.pyplot(fig2)

st.caption(f'最終更新: {pd.Timestamp.now().strftime("%Y/%m/%d %H:%M")}  |  データ: yfinance  |  キャッシュ: 1時間')
