"""Streamlit UI for the CFD Case Copilot.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from cfd_copilot.agent import AgentState, run_pipeline
from cfd_copilot.config import settings
from cfd_copilot.openfoam import find_bashrc, openfoam_available

st.set_page_config(page_title="CFD Case Copilot", page_icon="🌀", layout="wide")

EXAMPLES = [
    "Lid-driven cavity at 1 m/s with kinematic viscosity 0.01",
    "Turbulent channel flow at 10 m/s using k-omega SST, Re 50000",
    "Supersonic flow over a forward step at Mach 3, air at 300 K and 101325 Pa",
]

if "prompt" not in st.session_state:
    st.session_state.prompt = EXAMPLES[0]

st.title("🌀 CFD Case Copilot")
st.caption(
    "Describe a flow problem in plain English. The agent extracts a structured "
    "spec, generates a validated OpenFOAM case, meshes it, checks it, and runs the "
    "solver — self-correcting from OpenFOAM's own error output."
)

with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle("Use local LLM (Ollama)", value=True, help="Off = deterministic parser")
    out_dir = st.text_input("Output directory", value="runs")
    st.divider()
    of_ok = openfoam_available()
    st.write("OpenFOAM:", "✅ found" if of_ok else "❌ not found")
    if of_ok:
        st.caption(find_bashrc() or "")
    st.caption(f"Chat model: `{settings.chat_model}`  ·  Embed: `{settings.embed_model}`")
    st.divider()
    st.markdown(
        "**How to run**\n\n"
        "1. Pick or type a prompt\n"
        "2. Click **Run simulation**\n\n"
        "Prefer a richer UI? Run `cfd-copilot serve` and open "
        "http://127.0.0.1:8000 in your browser."
    )

st.subheader("Case prompt")
cols = st.columns(len(EXAMPLES))
for i, (c, ex) in enumerate(zip(cols, EXAMPLES)):
    label = ["Cavity", "Channel", "Forward step"][i]
    if c.button(label, help=ex, use_container_width=True):
        st.session_state.prompt = ex

st.text_area("Describe your case", key="prompt", height=90)

with st.expander("Advanced overrides (optional)"):
    st.caption(
        "Values set here always win over both the prompt extraction and the "
        "auto-derived defaults. Leave a field empty to keep the extracted/derived value."
    )
    c1, c2, c3 = st.columns(3)
    ov_end_time = c1.number_input("End time [s]", min_value=0.0, value=None,
                                  placeholder="auto", format="%.6f")
    ov_delta_t = c2.number_input("deltaT [s]", min_value=0.0, value=None,
                                 placeholder="auto", format="%.6g")
    ov_velocity = c3.number_input("Velocity [m/s]", min_value=0.0, value=None,
                                  placeholder="auto")
    c4, c5, c6 = st.columns(3)
    ov_length = c4.number_input("Domain length [m]", min_value=0.0, value=None,
                                placeholder="auto")
    ov_height = c5.number_input("Domain height [m]", min_value=0.0, value=None,
                                placeholder="auto")
    ov_nx = c6.number_input("Mesh nx (ny scales with it)", min_value=2, max_value=2000,
                            value=None, placeholder="auto", step=1)

run_full = st.button("Run simulation", type="primary", help="Generate, mesh, validate, and solve")
mesh_only = st.button("Mesh only", help="Generate + mesh + checkMesh (skip solver)")

if run_full or mesh_only:
    validate_only = mesh_only and not run_full
    overrides = {
        "control.end_time": ov_end_time or None,
        "control.delta_t": ov_delta_t or None,
        "velocity": ov_velocity or None,
        "geometry.length": ov_length or None,
        "geometry.height": ov_height or None,
        "mesh.nx": int(ov_nx) if ov_nx else None,
        "mesh.ny": int(ov_nx) if ov_nx else None,
    }
    state = AgentState(
        prompt=st.session_state.prompt,
        settings=settings,
        use_llm=use_llm,
        validate_only=validate_only,
        run_solver=not validate_only,
        out_dir=Path(out_dir),
        spec_overrides=overrides,
    )
    label = "Meshing and validating..." if validate_only else "Running full simulation..."
    with st.status(label, expanded=True) as status:
        run_pipeline(state)
        for line in state.log:
            st.write(("⚠️ " if line.startswith(("repair", "warning")) else "✅ ") + line)
        status.update(
            label=f"Done — status: {state.status}",
            state="complete" if state.status == "success" else "error",
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Extracted spec")
        if state.spec:
            if use_llm and state.spec.extraction_method == "rules":
                st.warning(
                    "LLM extraction failed (is Ollama running and the model pulled?). "
                    "Fell back to the rule-based parser."
                )
            st.caption(f"Parser used: `{state.spec.extraction_method}`")
            st.json(state.spec.model_dump(mode="json"))
    with right:
        st.subheader("Results")
        if state.validation:
            st.metric("Mesh cells", state.validation.n_cells or "—")
            st.write(state.validation.summary())
        if state.run:
            st.write(state.run.summary())
            if state.run.final_residuals:
                st.bar_chart(
                    {k: v for k, v in state.run.final_residuals.items()},
                    horizontal=True,
                )
        if state.case_dir and state.case_dir.is_dir():
            entries = sorted(
                p.name
                for p in state.case_dir.iterdir()
                if p.is_dir() and (p.name == "0" or p.name.replace(".", "", 1).isdigit())
            )
            if entries:
                st.write("Time folders:", ", ".join(entries))
            elif validate_only:
                st.info(
                    "Only the `0/` folder exists because the solver was skipped. "
                    "Click **Run simulation** to produce time-step results."
                )
            foam = state.case_dir / f"{state.spec.name}.foam"
            st.markdown("**Open in ParaView**")
            st.code(f"paraview {foam}", language="bash")
