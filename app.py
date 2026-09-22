import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from metrics import PERIODS, NEUTRAL, weekly_prices, calculate, state
from universe import SECTOR_ETFS, THEME_ETFS

st.set_page_config(page_title='セクターの流れ | RS Screener', page_icon='🌊', layout='wide')
st.markdown('''<style>.block-container{max-width:1250px;padding-top:2rem}h1{letter-spacing:-.04em} [data-testid="stMetric"]{background:rgba(120,140,160,.08);padding:16px;border-radius:12px}</style>''', unsafe_allow_html=True)
st.title('セクターの流れ')
st.caption('強さの位置と変化から、相場の移り変わりを読む')
ALL = {**SECTOR_ETFS, **THEME_ETFS}
COMMODITIES = {'GLD', 'SLV', 'USO', 'DBA'}
COLORS = ['#38bdf8','#a78bfa','#34d399','#fbbf24','#fb7185','#22d3ee','#f472b6','#a3e635','#fb923c','#818cf8','#94a3b8']
label = lambda s: f'{ALL.get(s,s)}（{s}）'

@st.cache_data(ttl=3600)
def load_data():
    raw = yf.download(list(ALL)+['QQQ'], period='3y', interval='1d', auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError('データを取得できませんでした。時間を置いて更新してください。')
    close = raw['Close']
    return close.reindex(columns=list(ALL)+['QQQ']), pd.Timestamp.now(tz='UTC')

c1,c2 = st.columns([3,1])
with c1:
    provisional = st.toggle('今週途中の動きも見る（暫定）', value=False)
with c2:
    if st.button('データを更新', use_container_width=True):
        load_data.clear()
try:
    with st.spinner('価格データを取得しています…'):
        daily, fetched = load_data()
        weekly, price_date, partial = weekly_prices(daily, provisional=provisional)
        periods, score, changes = calculate(weekly)
    if weekly['QQQ'].iloc[-1:].isna().any() or score.iloc[-1].dropna().empty:
        raise ValueError('基準日のデータが不足しています。データを更新してください。')
except Exception as exc:
    st.error(f'表示できません：{exc}')
    st.stop()

st.caption(f'比較基準：QQQ ｜ 価格基準日：{price_date:%Y/%m/%d} ｜ {"今週は暫定（日足終値まで）" if partial else "確定週足"} ｜ 取得：{fetched.tz_convert("Asia/Tokyo"):%m/%d %H:%M} 日本時間')
missing = [s for s in ALL if pd.isna(score[s].iloc[-1])]
if missing:
    st.warning(f'計算可能 {len(ALL)-len(missing)}/{len(ALL)}本。欠損・履歴不足：'+ '、'.join(map(label,missing)))


def render(symbols, key):
    available = [s for s in symbols if pd.notna(score[s].iloc[-1])]
    if not available:
        st.info('この分類のデータが不足しています。'); return
    latest = score[available].iloc[-1]
    delta = changes[available].iloc[-1]
    st.subheader('今週の変化' if not partial else '今週の変化 · 暫定')
    st.caption('総合RSの前週差。順位の上昇とは別に、数値自体の改善・悪化を表示します。')
    groups = [('↗ 強くなっている',delta[delta>NEUTRAL].sort_values(ascending=False)), ('→ 強さを維持',delta[(delta.abs()<=NEUTRAL)&(latest>=1)]), ('↘ 弱くなっている',delta[delta < -NEUTRAL].sort_values())]
    for col,(title,vals) in zip(st.columns(3),groups):
        with col:
            with st.container(border=True):
                st.markdown('**'+title+'**')
                if vals.empty: st.caption('該当なし')
                for sym,val in vals.head(3).items():
                    st.markdown(f'**{label(sym)}**')
                    st.caption(f'{state(latest[sym],val)} · {val:+.2f} pt')
    st.subheader('強弱マップ')
    st.caption('右ほどQQQに対して強く、上ほど前週から改善。線は直近6週、◆が最新。四象限を順番に回るとは限りません。')
    selected = st.multiselect('比較するセクター・テーマ', available, default=available if key=='sector' else latest.nlargest(5).index.tolist(), format_func=label, key='select_'+key)
    fig = go.Figure()
    for i,sym in enumerate(selected):
        x=(score[sym].tail(6)-1)*100; y=changes[sym].tail(6)
        fig.add_trace(go.Scatter(x=x,y=y,mode='lines+markers',name=label(sym),connectgaps=False,line=dict(color=COLORS[i%len(COLORS)],width=2),marker=dict(size=[5]*5+[12] if len(x)==6 else 8,symbol=['circle']*(len(x)-1)+['diamond']),customdata=x.index.strftime('%Y/%m/%d'),hovertemplate=label(sym)+'<br>週ラベル %{customdata}<br>相対強度 %{x:.2f} pt<br>前週差 %{y:.2f} pt<extra></extra>'))
    fig.add_hline(y=0,line_color='#94a3b8',line_dash='dot'); fig.add_vline(x=0,line_color='#94a3b8',line_dash='dot')
    for x,y,t in [(0,1,'弱いが改善'),(1,1,'強く、改善'),(0,0,'弱く、悪化'),(1,0,'強いが悪化')]:
        fig.add_annotation(x=x,y=y,xref='paper',yref='paper',text=t,showarrow=False,xanchor='left' if x==0 else 'right',font=dict(color='#94a3b8'))
    fig.update_layout(height=520,margin=dict(l=10,r=10,t=30,b=10),xaxis_title='総合RS − 1（×100） → 強い',yaxis_title='総合RSの前週差（×100） → 改善',legend=dict(orientation='h',y=-.22),hovermode='closest')
    st.plotly_chart(fig,use_container_width=True,key='map_'+key)
    st.subheader('時間別の強弱')
    rows=[]
    # Rank changes use only the common valid universe in both weeks.
    common=[s for s in available if pd.notna(score[s].iloc[-2])]
    now_rank=score[common].iloc[-1].rank(ascending=False,method='min')
    prev_rank=score[common].iloc[-2].rank(ascending=False,method='min')
    for sym in available:
        row={'セクター・テーマ':label(sym),'状態':state(latest[sym],delta[sym]),'前週差(pt)':delta[sym],'総合RS':latest[sym]}
        row.update({f'{p}週RS':periods[p][sym].iloc[-1] for p in PERIODS})
        row['4週騰落率(%)']=(weekly[sym].iloc[-1]/weekly[sym].iloc[-5]-1)*100
        row['順位変化']=prev_rank.get(sym,float('nan'))-now_rank.get(sym,float('nan'))
        row['52週最高終値比(%)']=(weekly[sym].iloc[-1]/weekly[sym].tail(52).max()-1)*100
        rows.append(row)
    frame=pd.DataFrame(rows).sort_values('前週差(pt)',ascending=False)
    rscols=['総合RS']+[f'{p}週RS' for p in PERIODS]
    def shade(v):
        return 'background-color:rgba(16,185,129,.22)' if v>=1 else 'background-color:rgba(244,63,94,.18)'
    st.dataframe(frame.style.map(shade,subset=rscols).format({**{c:'{:.3f}' for c in rscols},'前週差(pt)':'{:+.2f}','4週騰落率(%)':'{:+.2f}','順位変化':'{:+.0f}','52週最高終値比(%)':'{:.1f}'},na_rep='—'),hide_index=True,use_container_width=True,height=min(650,38*len(frame)+40))
    st.caption('RS 1＝QQQと同じ成績。緑＝上回る／赤＝下回る。順位変化は両週にデータがある同じ対象内で比較（＋は上昇）。')
    st.subheader('選択したセクターの相対推移')
    st.caption('26週前＝100にそろえた対QQQ推移。総合RSスコアとは別の指標です。')
    ratio=weekly[selected].div(weekly['QQQ'],axis=0).tail(27)
    normalized=ratio.div(ratio.iloc[0])*100
    chart=go.Figure()
    for i,sym in enumerate(selected):
        chart.add_trace(go.Scatter(x=normalized.index,y=normalized[sym],name=label(sym),line=dict(color=COLORS[i%len(COLORS)]),connectgaps=False))
    chart.add_hline(y=100,line_dash='dot'); chart.update_layout(height=420,margin=dict(l=10,r=10,t=10,b=10),legend=dict(orientation='h',y=-.2),yaxis_title='対QQQ相対推移',hovermode='x unified')
    st.plotly_chart(chart,use_container_width=True,key='trend_'+key)
    st.download_button('一覧をCSVで保存',frame.to_csv(index=False).encode('utf-8-sig'),file_name=f'rs_{key}_{price_date:%Y%m%d}.csv',mime='text/csv',key='csv_'+key)

for tab,syms,key in zip(st.tabs(['セクター','株式テーマ','コモディティ']),[SECTOR_ETFS,{s:n for s,n in THEME_ETFS.items() if s not in COMMODITIES},{s:ALL[s] for s in ALL if s in COMMODITIES}],['sector','theme','commodity']):
    with tab: render(syms,key)
with st.expander('指標の読み方・計算方法'):
    st.markdown('''- 各期間RS＝ETFの価格倍率 ÷ QQQの価格倍率。配当・分割調整後の価格を使用します。
- 総合RS＝4・13・26・52週RSの単純平均。元の重みを維持し、丸める前の値で計算します。
- 前週差＝（今週の総合RS − 前週の総合RS）×100。±0.20pt以内を横ばいとする表示上の基準で、売買シグナルではありません。
- マップの横軸は（総合RS − 1）×100、縦軸は前週差。独自の強弱マップで、JdK RRG指標ではありません。
- 確定週足は米国取引カレンダーの週最終取引日終値。暫定は今週の終了済み取引日までを使用し、日中の値は使用しません。
- 週ラベルは金曜日。祝日週の実際の価格基準日は画面上部に表示します。
- 53週のいずれかで価格が欠けるETFは除外し、過去価格で穴埋めしません。前週比較ができない場合は「比較データ不足」です。
- 過去推移は現在の対象ETFと取得した価格履歴から再計算します。当時の対象銘柄や改訂前データの記録ではありません。
- RSが上がっていてもETF自体が下落している場合があります。4週騰落率を併せて確認してください。RSから資金流入・流出は断定できません。''')
