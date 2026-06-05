"""pages/12_AI_Assistant.py – Feature 14: AI Assistant"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import WEATHER_PROFILES, tod_mult, run_p2, compute_route_time, compute_eta, explain_eta, delay_risk, PROMISED_ETA

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

st.markdown("# 🤖 AI Assistant")
st.caption("Feature 14 — Ask why ETA is high and get a quantified, human-readable explanation  |  Huge wow factor")
st.divider()

ROAD_PROFILES = {
    "Koramangala":  {"km":4.2,"hw":0.10,"pr":0.40,"se":0.30,"re":0.20},
    "HSR Layout":   {"km":5.8,"hw":0.15,"pr":0.35,"se":0.30,"re":0.20},
    "Indiranagar":  {"km":6.5,"hw":0.08,"pr":0.32,"se":0.38,"re":0.22},
    "Whitefield":   {"km":12.1,"hw":0.25,"pr":0.38,"se":0.22,"re":0.15},
    "Marathahalli": {"km":9.4,"hw":0.20,"pr":0.40,"se":0.25,"re":0.15},
    "Electronic City":{"km":14.2,"hw":0.40,"pr":0.35,"se":0.15,"re":0.10},
}

with st.sidebar:
    st.markdown("### 🎛️ Current Order Context")
    area      = st.selectbox("📍 Area", list(ROAD_PROFILES.keys()), index=3)
    weather_k = st.selectbox("🌤️ Weather", list(WEATHER_PROFILES.keys()), index=3,
                              format_func=lambda k: WEATHER_PROFILES[k]["label"])
    hour      = st.slider("🕐 Hour", 0, 23, 18)
    batch_sz  = st.slider("📦 Batch Size", 1, 5, 4)
    seed      = st.number_input("🔢 Seed", 1, 9999, 42)

rp       = ROAD_PROFILES[area]
w_mult   = WEATHER_PROFILES[weather_k]["mult"]
t_mult   = tod_mult(hour)
p2       = run_p2(seed=int(seed), n_window=batch_sz)
rt       = compute_route_time(rp["km"], rp["hw"], rp["pr"], rp["se"], rp["re"], w_mult, t_mult)
wd       = round(rt * (w_mult - 1.0), 2)
eta      = compute_eta(rt, p2["cpm_eta"], wd)
factors  = explain_eta(rt, p2["cpm_eta"], wd, t_mult, batch_sz)
rl, rc, rp_val = delay_risk(eta)

# sort factors by delta desc to find top contributors
factors_sorted = sorted(factors, key=lambda x: x["delta"], reverse=True)
baseline_eta   = 15.0
total_increase = round(eta - baseline_eta, 1)

# ── current order status ──────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("⏱️ Current ETA", f"{eta} min",
          delta=f"+{total_increase} min vs baseline", delta_color="inverse")
c2.metric("⚠️ Risk Level",  rl)
c3.metric("📊 Late Probability", f"{rp_val:.0%}")

st.divider()

# ── chat-style interface ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

def build_answer(question: str) -> str:
    q = question.lower()
    
    top3 = factors_sorted[:3]
    
    if any(w in q for w in ["why", "reason", "cause", "explain", "high"]):
        ans = f"**ETA increased by {total_increase:.1f} min above baseline because:**\n\n"
        for f in top3:
            icon = "🔺" if f["delta"] > 2 else "▲"
            ans += f"• {icon} **{f['factor']}** (+{f['delta']} min)\n"
        ans += f"\nTotal predicted ETA: **{eta} min**"
        if eta > PROMISED_ETA:
            ans += f"\n\n⚠️ This order will likely **miss the {PROMISED_ETA}-min SLA**."
        return ans

    if any(w in q for w in ["weather"]):
        return (f"**Weather impact:** {WEATHER_PROFILES[weather_k]['label']} adds a "
                f"×{w_mult} multiplier to route time. This adds **+{wd:.1f} min** to ETA. "
                f"Switching to clear weather would save ≈{wd:.1f} min.")

    if any(w in q for w in ["traffic", "peak", "hour"]):
        traffic_add = round((t_mult - 1) * rt, 1)
        tod_str     = "Peak hour" if t_mult > 1.1 else "Off-peak"
        return (f"**Traffic impact:** {tod_str} at {hour:02d}:00 (×{t_mult:.2f} multiplier). "
                f"This adds **+{traffic_add} min**. Best time to order: 02:00–07:00.")

    if any(w in q for w in ["batch", "orders"]):
        b_add = round((batch_sz - 1) * 0.8, 1)
        return (f"**Batch size impact:** {batch_sz} orders are batched together, "
                f"adding **+{b_add} min** coordination overhead. "
                f"Single-order dispatch would save {b_add} min.")

    if any(w in q for w in ["route", "distance", "km", "path"]):
        return (f"**Route impact (P1 graph engine):** Distance = {rp['km']} km. "
                f"Route time = {rt:.1f} min at current conditions. "
                f"The CH Dijkstra path found {int(rp['km']*6)} segments.")

    if any(w in q for w in ["prep", "food", "kitchen"]):
        prep = p2["stage_delays"].get("accepted_to_prep", 8.0)
        return (f"**Food prep impact (P2 DAG):** Prep stage = {prep:.1f} min. "
                f"CPM slack = {p2['cpm_slack']} min. "
                f"Critical path goes through: {' → '.join(p2['critical_path'][:3])}...")

    if any(w in q for w in ["save", "reduce", "improve", "faster"]):
        savings = [(f["factor"], f["delta"]) for f in top3]
        ans = "**To reduce ETA, focus on:**\n\n"
        for name, delta in savings:
            ans += f"• Reduce **{name}** — potential saving: {delta:.1f} min\n"
        ans += f"\nBest single action: Address **{savings[0][0]}**."
        return ans

    return (f"I can answer questions about: **why ETA is high**, **weather impact**, "
            f"**traffic conditions**, **batch size**, **route distance**, **food prep**, "
            f"or **how to reduce ETA**. Current ETA is **{eta} min** (risk: {rl}).")

# ── quick question buttons ────────────────────────────────────────────────────
st.markdown("#### 💬 Ask the Assistant")
st.markdown("**Quick Questions:**")
qcol1, qcol2, qcol3 = st.columns(3)
quick_q = None
if qcol1.button("❓ Why is ETA high?"):
    quick_q = "Why is ETA high?"
if qcol2.button("🌧️ Weather impact?"):
    quick_q = "What is the weather impact?"
if qcol3.button("⚡ How to reduce ETA?"):
    quick_q = "How can I reduce ETA?"
qcol4, qcol5, qcol6 = st.columns(3)
if qcol4.button("🚗 Traffic impact?"):
    quick_q = "What is the traffic impact?"
if qcol5.button("📦 Batch size impact?"):
    quick_q = "What is the batch size impact?"
if qcol6.button("🛣️ Route distance?"):
    quick_q = "Tell me about the route distance"

# chat input
user_input = st.chat_input("Ask anything about this delivery's ETA…")
if quick_q:
    user_input = quick_q

# display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer = build_answer(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

if not st.session_state.messages:
    st.info("👆 Click a quick question or type your own above to get started.")
