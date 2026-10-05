"""Streamlit dashboard: live risk signals (engine) and event-driven portfolio stress tests (Module B)."""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

from ripple.engine.regions import REGIONS
from ripple.schemas import EventType, RiskSignal
from ripple.stress_test.portfolio import load_portfolio
from ripple.stress_test.scenarios import build_scenario, load_library
from ripple.stress_test.worker import stress_result

API_URL = os.getenv("API_URL", "http://localhost:8000")
REFRESH_SECONDS = 3
HIGH_IMPACT = float(os.getenv("STRESS_IMPACT_THRESHOLD", "7"))

# Fixed entity -> colour (categorical slots in validated order), so colour follows the entity, never its rank.
ENTITY_COLORS = {
    "FB": "#2a78d6",
    "AAPL": "#eb6834",
    "GOOGL": "#1baf7a",
    "AMZN": "#eda100",
    "NFLX": "#e87ba4",
    "MSFT": "#008300",
    "MARKET": "#4a3aa7",
    "DIS": "#e34948",
}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = "#2a78d6"
GAIN, LOSS = "#2a78d6", "#e34948"  # diverging poles (validated pair), never status colours
STATUS = [(8.5, "Critical", "#d03b3b", "▲▲"), (7.0, "Serious", "#ec835a", "▲"), (0.0, "Watch", "#fab219", "●")]

st.set_page_config(page_title="Ripple", layout="wide")


def chart_theme(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=MUTED, titleColor=INK_2, labelFontSize=11
        )
        .configure_legend(labelColor=INK_2, titleColor=INK_2, orient="top")
        .properties(background="transparent")
    )


def fetch(path: str, **params):
    return requests.get(f"{API_URL}{path}", params=params, timeout=5).json()


def status_label(impact: float) -> str:
    for floor, label, _, icon in STATUS:
        if impact >= floor:
            return f"{icon} {label}"
    return ""


def sentiment_chart(df: pd.DataFrame, entities: list[str]) -> alt.Chart:
    data = df[df.entity.isin(entities)].copy()
    data["day"] = data.published_at.dt.floor("D")
    daily = data.groupby(["day", "entity"], as_index=False).agg(
        sentiment=("sentiment_score", "mean"), signals=("signal_id", "count")
    )
    color = alt.Color(
        "entity:N",
        title=None,
        scale=alt.Scale(domain=entities, range=[ENTITY_COLORS[e] for e in entities]),
    )
    hover = alt.selection_point(fields=["day"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(daily).encode(x=alt.X("day:T", title=None, axis=alt.Axis(format="%d %b", grid=False)))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=AXIS, strokeWidth=1).encode(y="y:Q")
    lines = base.mark_line(strokeWidth=2, interpolate="monotone").encode(
        y=alt.Y("sentiment:Q", title="Mean sentiment", scale=alt.Scale(domain=[-1, 1])), color=color
    )
    points = base.mark_point(size=60, filled=True, opacity=0).encode(
        y="sentiment:Q",
        color=color,
        tooltip=[
            alt.Tooltip("day:T", title="Day", format="%d %b %Y"),
            alt.Tooltip("entity:N", title="Entity"),
            alt.Tooltip("sentiment:Q", title="Sentiment", format="+.2f"),
            alt.Tooltip("signals:Q", title="Signals"),
        ],
    ).add_params(hover)
    rule = base.mark_rule(color=MUTED, strokeWidth=1).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    shown = points.encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    return chart_theme((zero + lines + rule + shown).properties(height=300))


def event_mix_chart(df: pd.DataFrame) -> alt.Chart:
    counts = df.groupby("event_type", as_index=False).size().rename(columns={"size": "signals"})
    order = counts.sort_values("signals", ascending=False).event_type.tolist()
    bars = (
        alt.Chart(counts)
        .mark_bar(color=SEQ_BLUE, cornerRadiusEnd=4, height=14)
        .encode(
            x=alt.X("signals:Q", title=None, axis=alt.Axis(grid=True, tickCount=4)),
            y=alt.Y("event_type:N", sort=order, title=None, axis=alt.Axis(labelColor=INK_2, labelLimit=160)),
            tooltip=[alt.Tooltip("event_type:N", title="Event"), alt.Tooltip("signals:Q", title="Signals")],
        )
    )
    labels = bars.mark_text(align="left", dx=4, color=INK_2, fontSize=11).encode(text="signals:Q")
    return chart_theme((bars + labels).properties(height=300))


def evidence_card(row: pd.Series) -> None:
    breakdown = row.evidence.get("impact_breakdown", {})
    st.markdown(f"**{row.entity}** · {row.event_type} · {status_label(row.impact_score)} · impact **{row.impact_score:.1f}**")
    model = row.evidence.get("model") or "rules+finbert"
    st.caption(f"{row.published_at:%d %b %Y %H:%M} · {row.source} · analysed by {model}")
    st.write(row.headline)
    cols = st.columns(4)
    cols[0].metric("Sentiment", f"{row.sentiment_score:+.2f}")
    if "p_material_move" in breakdown:  # calibrated impact model
        cols[1].metric("P(abnormal move ≥ 2σ)", f"{breakdown['p_material_move']:.0%}")
        cols[2].metric("Severity (model)", f"{breakdown.get('severity', 0):.2f}")
        cols[3].metric("Buzz (abnormal attention)", f"{breakdown.get('buzz', 0):.2f}")
        drivers = {k.removeprefix("contrib_"): v for k, v in breakdown.items() if k.startswith("contrib_")}
        st.caption("What drove the score (log-odds): " + " · ".join(
            f"{k} **{v:+.2f}**" for k, v in sorted(drivers.items(), key=lambda kv: -abs(kv[1]))))
    else:
        cols[1].metric("Severity × confidence", f"{breakdown.get('severity', 0):.2f}")
        cols[2].metric("Buzz (abnormal attention)", f"{breakdown.get('buzz', 0):.2f}")
        cols[3].metric("Source credibility", f"{breakdown.get('source_credibility', 0):.2f}")
    phrases = row.evidence.get("key_phrases", [])
    if phrases:
        st.caption("Key phrases: " + " · ".join(f"`{p}`" for p in phrases))
    for url in row.evidence.get("source_urls", []):
        st.caption(f"[Source]({url})")


st.title("Ripple")
st.caption("Real-time NLP risk signals from news and social media, and the portfolio stress tests they trigger")


@st.fragment(run_every=REFRESH_SECONDS)
def live_view() -> None:
    try:
        stats = fetch("/stats")
        signals = fetch("/signals", limit=10_000)
    except requests.RequestException as e:
        st.error(f"API not reachable at {API_URL}: {e}")
        return

    engine = stats.get("engine", {})
    received, noise = engine.get("received", 0), engine.get("filtered_noise", 0)
    clock = pd.to_datetime(stats.get("replay_clock")) if stats.get("replay_clock") else None

    df = pd.DataFrame(signals)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df.published_at)
    high = df[df.impact_score >= HIGH_IMPACT] if not df.empty else df

    k = st.columns(4)
    k[0].metric("Replay clock", f"{clock:%d %b %Y %H:%M}" if clock is not None else "—")
    k[1].metric("Documents ingested", f"{received:,}")
    k[2].metric("Noise filtered", f"{noise / received:.0%}" if received else "—", help="Spam / non-financial social posts removed")
    k[3].metric(f"High-impact signals (≥ {HIGH_IMPACT:g})", f"{len(high):,}")

    if df.empty:
        st.info("Waiting for signals… the ingestor replays Jul–Aug 2018 news and tweets.")
        return

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Sentiment by entity")
        available = [e for e in ENTITY_COLORS if e in set(df.entity)]
        chosen = st.multiselect(
            "Entities", available, default=available[:3], max_selections=4, key="entities", label_visibility="collapsed"
        )
        if chosen:
            st.altair_chart(sentiment_chart(df, chosen), use_container_width=True)
    with right:
        st.subheader("Event mix")
        st.altair_chart(event_mix_chart(df), use_container_width=True)

    st.subheader("High-impact alerts")
    if high.empty:
        st.caption("No high-impact signals yet.")
        return
    alerts = high.sort_values(["impact_score", "published_at"], ascending=False).head(50).reset_index(drop=True)
    table = alerts.assign(status=alerts.impact_score.map(status_label))[
        ["status", "published_at", "entity", "event_type", "sentiment_score", "impact_score", "source", "headline"]
    ]
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "published_at": st.column_config.DatetimeColumn("Time", format="DD MMM HH:mm"),
            "sentiment_score": st.column_config.NumberColumn("Sentiment", format="%+.2f"),
            "impact_score": st.column_config.NumberColumn("Impact", format="%.1f"),
            "event_type": "Event",
            "entity": "Entity",
            "status": "Status",
            "source": "Source",
            "headline": st.column_config.TextColumn("Headline", width="large"),
        },
    )
    with st.expander("Evidence card: why did the engine score the top alert this way?", expanded=True):
        evidence_card(alerts.iloc[0])



def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x) / 1e9:,.2f}bn" if abs(x) >= 1e9 else f"{sign}${abs(x) / 1e6:,.1f}m"


def pnl_bars(values: dict[str, float], title: str) -> alt.Chart:
    data = pd.DataFrame({"bucket": list(values), "pnl": [v / 1e6 for v in values.values()]})
    data = data[data.pnl.abs() >= 0.05]
    data["direction"] = ["Gain" if v >= 0 else "Loss" for v in data.pnl]
    data["label"] = [f"{v:+,.1f}" for v in data.pnl]
    order = data.sort_values("pnl").bucket.tolist()
    base = alt.Chart(data).encode(
        y=alt.Y("bucket:N", sort=order, title=None, axis=alt.Axis(labelColor=INK_2, labelLimit=160)),
        tooltip=[alt.Tooltip("bucket:N", title=title), alt.Tooltip("pnl:Q", title="P&L ($m)", format="+,.1f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=14).encode(
        x=alt.X("pnl:Q", title="P&L ($m)", axis=alt.Axis(grid=True, tickCount=5)),
        color=alt.Color("direction:N", title=None, scale=alt.Scale(domain=["Gain", "Loss"], range=[GAIN, LOSS])),
    )
    gains = base.transform_filter("datum.pnl >= 0").mark_text(align="left", dx=4, color=INK_2, fontSize=11)
    losses = base.transform_filter("datum.pnl < 0").mark_text(align="right", dx=-4, color=INK_2, fontSize=11)
    labels = [m.encode(x="pnl:Q", text="label:N") for m in (gains, losses)]
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=AXIS).encode(x="x:Q")
    return chart_theme(alt.layer(zero, bars, *labels).properties(height=max(120, 34 * len(data))))


def stress_card(r: dict) -> None:
    t = r["trigger"]
    st.markdown(
        f"**Trigger:** {t['entity']} · {t['event_type']} · {status_label(t['impact_score'])} · "
        f"impact **{t['impact_score']:.1f}** · {pd.to_datetime(t['published_at']):%d %b %Y %H:%M}"
    )
    st.write(f"“{t['headline']}”")
    st.markdown(f"**Scenario:** {r['scenario_title']} · severity ×{r['severity_multiplier']:.0%}")
    st.caption(r["narrative"])

    k = st.columns(4)
    k[0].metric("Portfolio value", money(r["value_after"]), f"{money(r['pnl_total'])} from {money(r['value_before'])}")
    k[1].metric("CET1 ratio", f"{r['cet1_ratio_after']:.2%}",
                f"{100 * (r['cet1_ratio_after'] - r['cet1_ratio_before']):+.2f} pp from {r['cet1_ratio_before']:.1%}")
    k[2].metric("Expected loss (loans)", money(r["expected_loss_after"]),
                money(r["expected_loss_after"] - r["expected_loss_before"]), delta_color="inverse")
    k[3].metric("Credit RWA", money(r["rwa_after"]), money(r["rwa_after"] - r["rwa_before"]), delta_color="inverse")

    left, right = st.columns(2)
    with left:
        st.markdown("**P&L by asset class**")
        st.altair_chart(pnl_bars(r["pnl_by_asset_class"], "Asset class"), use_container_width=True)
    with right:
        st.markdown("**P&L by region**")
        st.altair_chart(pnl_bars(r["pnl_by_region"], "Region"), use_container_width=True)

    st.markdown("**Largest position impacts**")
    top = pd.DataFrame(r["top_positions"])
    if not top.empty:
        top["pnl"] = top.pnl / 1e6
        st.dataframe(
            top[["counterparty", "asset_class", "region", "pnl"]], hide_index=True, use_container_width=True,
            column_config={"counterparty": "Position", "asset_class": "Asset class", "region": "Region",
                           "pnl": st.column_config.NumberColumn("P&L ($m)", format="%+.1f")},
        )
    with st.expander("Shocks applied and assumptions"):
        shocks = [
            {"factor": f, "target": k, "shock": f"{v:+.1%}" if f in ("equity", "fx") else f"{v:+.0f} bp"}
            for f, d in r["shocks"].items() for k, v in d.items()
        ]
        st.dataframe(pd.DataFrame(shocks), hide_index=True, use_container_width=True)
        for a in r["assumptions"]:
            st.caption(f"• {a}")
        st.caption(
            "• Sensitivity-based revaluation (duration/convexity, DV01, spread DV01, delta); loan losses as "
            "expected-loss provisions with PDs stressed by the spread move; Basel IRB credit RWA; CET1 capital "
            "set at 13% of baseline RWA; losses pre-tax."
        )


@st.cache_resource
def stress_inputs():
    return load_portfolio(), load_library()


def what_if() -> None:
    book, library = stress_inputs()
    companies = sorted(t for t in book.ticker.dropna().unique() if t != "MARKET")
    c = st.columns([2, 2, 2, 3])
    event = c[0].selectbox("Event type", [e.value for e in EventType if e != EventType.OTHER], key="wi_event")
    entity = c[1].selectbox("Entity", ["MARKET", *REGIONS, *companies], key="wi_entity")
    impact = c[2].slider("Impact score", 7.0, 10.0, 9.0, 0.1, key="wi_impact")
    headline = c[3].text_input("Headline (optional)", "", key="wi_headline")
    sig = RiskSignal(
        doc_id="what-if", published_at=pd.Timestamp.now(tz="UTC"), source="news", entity=entity,
        sentiment_score=-0.9, event_type=event, event_confidence=1.0, impact_score=impact,
        headline=headline or f"What-if: {event} event, {entity}",
    )
    stress_card(stress_result(sig, build_scenario(sig, library), book).model_dump(mode="json"))


@st.fragment(run_every=REFRESH_SECONDS)
def stress_view() -> None:
    try:
        results = fetch("/stress", limit=200)
    except requests.RequestException as e:
        st.error(f"API not reachable at {API_URL}: {e}")
        return
    if not results:
        st.info(f"No stress test yet. A signal with impact ≥ {HIGH_IMPACT:g} triggers one; try the What-if tab meanwhile.")
        return
    history = pd.DataFrame([{
        "time": pd.to_datetime(r["trigger"]["published_at"]), "entity": r["trigger"]["entity"],
        "event": r["trigger"]["event_type"], "impact": r["trigger"]["impact_score"], "scenario": r["scenario_title"],
        "pnl": r["pnl_total"] / 1e6, "cet1": 100 * r["cet1_ratio_after"], "headline": r["trigger"]["headline"],
    } for r in results])
    k = st.columns(3)
    k[0].metric("Stress tests triggered", f"{len(results):,}")
    k[1].metric("Worst P&L", f"{history.pnl.min():+,.1f}m")
    k[2].metric("Lowest CET1 ratio", f"{history.cet1.min():.2f}%")
    labels = [f"{h.time:%d %b %H:%M} · {h.entity} · {h.event} · {h.pnl:+,.1f}m" for h in history.itertuples()]
    idx = st.selectbox("Stress test (latest first)", range(len(results)), format_func=labels.__getitem__, key="stress_pick")
    stress_card(results[idx])
    with st.expander(f"All {len(results)} stress tests"):
        st.dataframe(history, hide_index=True, use_container_width=True, column_config={
            "time": st.column_config.DatetimeColumn("Time", format="DD MMM HH:mm"),
            "impact": st.column_config.NumberColumn("Impact", format="%.1f"),
            "pnl": st.column_config.NumberColumn("P&L ($m)", format="%+.1f"),
            "cet1": st.column_config.NumberColumn("CET1 after (%)", format="%.2f"),
            "headline": st.column_config.TextColumn("Headline", width="large"),
        })


signals_tab, stress_tab, what_if_tab = st.tabs(["Risk signals", "Stress testing (Module B)", "What-if scenario"])
with signals_tab:
    live_view()
with stress_tab:
    stress_view()
with what_if_tab:
    st.caption("Pick an event and see how the portfolio would fare. Same scenario library and pricing as the live tests.")
    what_if()
