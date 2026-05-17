"""A股主升浪五维共振 -- Streamlit 可视化仪表盘.

运行:
    pip install -r requirements.txt streamlit
    streamlit run app.py

离线演示模式（默认）无需 akshare/网络，使用内置合成场景；
选择「AkShare 实盘」需已 `pip install -U akshare`。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ashare_main_wave import (position_plan, score_stock, stop_loss_price,
                              take_profit_action)
from ashare_main_wave.demo_data import SCENARIOS, DemoProvider
from ashare_main_wave.screener import screen

st.set_page_config(page_title="A股主升浪 五维共振", layout="wide")

GRADE_COLOR = {"A+": "#0a8f4f", "A": "#2bb673", "B": "#e0a800",
               "C": "#e8590c", "D": "#c92a2a"}
SCENARIO_LABEL = {
    "textbook": "教科书级 A（满五维共振）",
    "b_grade": "B 级（结构强但无热点共振）",
    "d_grade": "D 级（弱而无否决）",
}
DEMO_UNIVERSE = {"600519": "textbook", "600001": "b_grade",
                 "600002": "d_grade"}


class _MultiDemoProvider(DemoProvider):
    """Routes each code in the screener demo to its own scenario."""

    def __init__(self):
        super().__init__("textbook")

    def _for(self, symbol):
        return DemoProvider(DEMO_UNIVERSE.get(str(symbol), "d_grade"))

    def kline(self, symbol, period, start, end, adjust="qfq"):
        return self._for(symbol).kline(symbol, period, start, end, adjust)

    def _proxy(self, symbol, name):
        return getattr(self._for(symbol), name)(symbol)

    def lhb(self, s, *a): return self._for(s).lhb(s)
    def block_trade(self, s, *a): return self._for(s).block_trade(s)
    def northbound(self, s, *a): return self._for(s).northbound(s)
    def institution_survey(self, s, *a): return self._for(s).institution_survey(s)
    def margin(self, s, *a): return self._for(s).margin(s)
    def fund_flow(self, s, *a): return self._for(s).fund_flow(s)
    def chip(self, s, *a): return self._for(s).chip(s)
    def concepts_of(self, s): return self._for(s).concepts_of(s)
    def concept_rank(self): return DemoProvider("textbook").concept_rank()
    def basic_info(self, s): return self._for(s).basic_info(s)


def _get_provider(mode: str):
    if mode == "AkShare 实盘":
        from ashare_main_wave import AkShareProvider
        return AkShareProvider()
    return None  # demo handled per-call


def _grade_badge(grade: str) -> str:
    c = GRADE_COLOR.get(grade, "#666")
    return (f"<span style='background:{c};color:#fff;padding:4px 14px;"
            f"border-radius:6px;font-size:1.4rem;font-weight:700'>"
            f"{grade}</span>")


# --------------------------------------------------------------------------
st.title("A股主升浪起涨点 · 五维共振评分")
st.caption("⚠️ 阈值为历史经验值，仅供研究，不构成任何投资建议。")

with st.sidebar:
    st.header("数据源")
    mode = st.radio("模式", ["离线演示", "AkShare 实盘"], index=0)
    st.divider()
    st.header("说明")
    st.markdown(
        "- **离线演示**：内置合成场景，无需网络/akshare\n"
        "- **AkShare 实盘**：需 `pip install -U akshare`\n"
        "- 维度：月线盘整 / 月线突破 / 日线趋势 / 主力行为 / 热点共振\n"
        "- 七大一票否决任一触发即归 D 级")

tab_score, tab_screen = st.tabs(["📊 单股评分", "🔍 池筛选"])

with tab_score:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        if mode == "离线演示":
            scen = st.selectbox("演示场景", SCENARIOS,
                                format_func=lambda s: SCENARIO_LABEL[s])
            symbol = st.text_input("代码（演示仅作标签）", "600519")
            date = st.text_input("评估日 YYYYMMDD", "20240430")
            prov = DemoProvider(scen)
            kwargs = prov.score_kwargs()
            st.caption(f"D5 输入: {kwargs or '（无定性输入）'}")
        else:
            symbol = st.text_input("股票代码", "300308")
            date = st.text_input("评估日 YYYYMMDD", "20230430")
            c1, c2, c3 = st.columns(3)
            catalysts = c1.number_input("催化剂数", 0, 9, 2)
            leader = c2.number_input("板块龙头排名", 0, 50, 1)
            limitups = c3.number_input("板块5日涨停数", 0, 99, 12)
            prov = _get_provider(mode)
            kwargs = dict(catalysts=int(catalysts),
                          leader_rank=int(leader) or None,
                          limitups_in_sector=int(limitups) or None)
        run = st.button("▶ 评分", type="primary", use_container_width=True)

    with col_out:
        if run:
            try:
                res = score_stock(symbol, date, prov, **kwargs)
            except Exception as exc:  # noqa: BLE001
                st.error(f"评分失败：{exc}")
            else:
                m1, m2 = st.columns(2)
                m1.metric("总分", f"{res.total:.2f} / 100")
                m2.markdown("**等级**　" + _grade_badge(res.grade),
                            unsafe_allow_html=True)
                if res.vetoes:
                    st.error("🚫 一票否决：\n\n- " + "\n- ".join(res.vetoes))

                dim_df = pd.DataFrame([
                    {"维度": k, "贡献分": round(v.contribution, 2),
                     "维度分(0-10)": round(v.score, 2)}
                    for k, v in res.dimensions.items()
                ]).set_index("维度")
                st.subheader("各维度贡献")
                st.bar_chart(dim_df["贡献分"])

                with st.expander("子指标明细 / 关键指标", expanded=False):
                    for k, v in res.dimensions.items():
                        st.markdown(f"**{k}** · score={v.score:.2f} · "
                                    f"contrib={v.contribution:.2f}"
                                    + (f" · 🚫{v.veto}" if v.veto else ""))
                        st.write({"subscores": v.subscores,
                                  "metrics": v.metrics})

                st.subheader("风控方案（spec §7）")
                pp = position_plan(res.grade)
                r1, r2 = st.columns(2)
                r1.write({"等级": pp.grade, "初始仓位": pp.initial,
                          "单票上限": pp.max,
                          "加仓金字塔": [list(x) for x in pp.pyramid]})
                last = float(res.dimensions["d3"].metrics.get(
                    "last_close", 0) or 0)
                if last:
                    sl = stop_loss_price(last, cap_size="small",
                                         ma20=res.dimensions["d3"].metrics
                                         .get("ma20"))
                    r2.write({"参考现价": round(last, 2),
                              "止损(取最高/最紧)": round(
                                  sl["effective_stop"], 2),
                              "候选": {k: round(v, 2) for k, v in
                                       sl["candidates"].items()}})
                tp = pd.DataFrame([
                    {"浮盈": f"+{int(p*100)}%",
                     "决策": take_profit_action(
                         p, below_ma10=(p in (0.35, 0.65)))["action"]}
                    for p in (0.10, 0.35, 0.65, 1.20)
                ])
                st.caption("分段移动止盈示例")
                st.dataframe(tp, hide_index=True, use_container_width=True)
        else:
            st.info("填写左侧参数后点击「评分」。")

with tab_screen:
    st.markdown("对一组代码批量评分并分桶（狙击池 / 观察池 / 排除）。")
    if mode == "离线演示":
        st.caption("演示宇宙：600519→A，600001→B，600002→D")
        codes = st.text_input("代码（逗号分隔）",
                              ",".join(DEMO_UNIVERSE)).split(",")
        sdate = st.text_input("日期", "20240430", key="sd")
        sprov = _MultiDemoProvider()
    else:
        codes = st.text_input("代码（逗号分隔）",
                              "300308,688256,600150").split(",")
        sdate = st.text_input("日期", "20230430", key="sd2")
        sprov = _get_provider(mode)

    if st.button("▶ 运行筛选", type="primary"):
        codes = [c.strip() for c in codes if c.strip()]
        with st.spinner("评分中…"):
            sr = screen(sprov, sdate, universe=codes)
        c1, c2, c3 = st.columns(3)
        c1.metric("狙击池 A/A+", len(sr.sniper_pool))
        c2.metric("观察池 B", len(sr.watch_pool))
        c3.metric("排除 C/D", sr.rejected)

        def _tbl(pool):
            return pd.DataFrame([{"代码": s.symbol, "总分": s.total,
                                  "等级": s.grade,
                                  "否决": "；".join(s.vetoes) or "—"}
                                 for s in pool])
        if sr.sniper_pool:
            st.subheader("🎯 狙击池")
            st.dataframe(_tbl(sr.sniper_pool), hide_index=True,
                         use_container_width=True)
        if sr.watch_pool:
            st.subheader("👀 观察池")
            st.dataframe(_tbl(sr.watch_pool), hide_index=True,
                         use_container_width=True)
        if sr.errors:
            st.warning(f"{len(sr.errors)} 个标的评分异常（已跳过）")
