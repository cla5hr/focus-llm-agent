"""The agentic loop: prompt -> spec -> case -> validate -> (repair) -> run.

The orchestration is written as plain, testable node functions in
:func:`run_pipeline`. A real LangGraph ``StateGraph`` is also exposed via
:func:`build_graph` for the agentic-tooling story; both share the same nodes.

Self-correction is grounded: the agent reacts to *OpenFOAM's own* error output
(from the validator / runner), not to the LLM's imagination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from cfd_copilot.config import Settings, settings as default_settings
from cfd_copilot.generator import generate_case
from cfd_copilot.llm import extract_spec
from cfd_copilot.runner import RunReport, run_solver
from cfd_copilot.schema import CaseSpec, MeshSpec
from cfd_copilot.validator import ValidationReport, validate_case


@dataclass
class AgentState:
    prompt: str
    settings: Settings = default_settings
    use_llm: bool = True
    validate_only: bool = False
    run_solver: bool = True
    out_dir: Path = field(default_factory=lambda: Path("runs"))
    # Explicit parameter overrides applied after extraction, keyed by dotted
    # path into CaseSpec (e.g. {"control.end_time": 2.0, "geometry.height": 2.0}).
    # This gives UIs a reliable knob that does not depend on NL extraction.
    spec_overrides: Optional[Dict[str, object]] = None

    spec: Optional[CaseSpec] = None
    case_dir: Optional[Path] = None
    validation: Optional[ValidationReport] = None
    run: Optional[RunReport] = None

    attempts: int = 0
    log: List[str] = field(default_factory=list)
    status: str = "pending"  # pending|success|failed

    def note(self, msg: str) -> None:
        self.log.append(msg)


# --------------------------------------------------------------------- nodes
def node_extract(state: AgentState) -> AgentState:
    state.spec = extract_spec(state.prompt, state.settings, use_llm=state.use_llm)
    if state.spec_overrides:
        state.spec = _apply_overrides(state.spec, state.spec_overrides)
        applied = ", ".join(f"{k}={v}" for k, v in state.spec_overrides.items() if v is not None)
        if applied:
            state.note(f"Applied user overrides: {applied}.")
    state.note(
        f"Interpreted as a '{state.spec.case_type.value}' case "
        f"(parser={state.spec.extraction_method}, "
        f"solver={state.spec.solver.value}, U={state.spec.velocity:g} m/s, "
        f"turbulence={state.spec.turbulence.value}, Re~{state.spec.reynolds():.3g})."
    )
    return state


def _apply_overrides(spec: CaseSpec, overrides: Dict[str, object]) -> CaseSpec:
    """Set dotted-path overrides on a spec and re-validate it.

    User-provided values win over auto-derived ones because the schema only
    derives fields that are unset -- an explicit value is never recomputed.
    """
    data = spec.model_dump()
    touched = False
    for path, value in overrides.items():
        if value is None:
            continue
        keys = path.split(".")
        d = data
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
        touched = True
    if not touched:
        return spec
    # Keep write_interval consistent if end_time was overridden but
    # write_interval was not.
    if overrides.get("control.end_time") is not None and overrides.get("control.write_interval") is None:
        data["control"]["write_interval"] = float(overrides["control.end_time"]) / 5.0
    return CaseSpec(**data)


def node_generate(state: AgentState) -> AgentState:
    assert state.spec is not None
    state.case_dir = generate_case(state.spec, state.out_dir)
    state.note(f"Generated case at {state.case_dir}")
    return state


def node_validate(state: AgentState) -> AgentState:
    assert state.case_dir is not None
    state.validation = validate_case(state.case_dir)
    state.note(state.validation.summary())
    for w in state.validation.warnings:
        state.note(f"warning: {w}")
    if state.validation.ok:
        if state.validate_only or not state.run_solver:
            state.status = "success"
    elif state.attempts >= state.settings.max_repair_attempts:
        state.status = "failed"
    return state


def node_run(state: AgentState) -> AgentState:
    assert state.spec is not None and state.case_dir is not None
    state.run = run_solver(state.spec, state.case_dir)
    state.note(state.run.summary())
    if state.run.ok:
        state.status = "success"
    elif state.attempts >= state.settings.max_repair_attempts:
        state.status = "failed"
    return state


def node_repair(state: AgentState) -> AgentState:
    """Apply escalating, deterministic fixes based on the failure observed."""
    assert state.spec is not None
    state.attempts += 1
    spec = state.spec

    failed_validation = state.validation is not None and not state.validation.ok
    failed_run = state.run is not None and not state.run.ok

    if failed_validation:
        # Mesh problems: fall back to a conservative, conformal mesh.
        spec.mesh = MeshSpec(nx=40, ny=40, nz=1, grading=1.0)
        state.note("repair: reset mesh to a safe 40x40 conformal grid.")
    elif failed_run:
        if spec.steady:
            spec.control.max_iterations = min(spec.control.max_iterations * 2, 5000)
            state.note(
                f"repair: solver did not converge; increasing iterations to "
                f"{spec.control.max_iterations}."
            )
        else:
            spec.control.delta_t = spec.control.delta_t / 5.0
            state.note(
                f"repair: solver unstable; reducing deltaT to {spec.control.delta_t:g}."
            )
    # Reset stale reports so routing re-evaluates from scratch.
    state.validation = None
    state.run = None
    return state


# ------------------------------------------------------------------- routing
def _should_repair(state: AgentState) -> bool:
    if state.validation is not None and not state.validation.ok:
        return True
    if state.run is not None and not state.run.ok:
        return True
    return False


def run_pipeline(state: AgentState) -> AgentState:
    """Execute the full agent loop without requiring LangGraph."""
    node_extract(state)

    max_attempts = state.settings.max_repair_attempts
    while True:
        node_generate(state)
        node_validate(state)

        if not state.validation.ok:
            if state.attempts < max_attempts:
                node_repair(state)
                continue
            state.status = "failed"
            return state

        if state.validate_only or not state.run_solver:
            state.status = "success"
            return state

        node_run(state)
        if state.run.ok:
            state.status = "success"
            return state
        if state.attempts < max_attempts:
            node_repair(state)
            continue
        state.status = "failed"
        return state


# ------------------------------------------------------- optional LangGraph
def build_graph(checkpointer=None):
    """Build an equivalent LangGraph ``StateGraph`` (optional dependency).

    Returns a compiled graph whose state is the :class:`AgentState` dataclass.
    Provided for the agentic-tooling story; ``run_pipeline`` is the default path.
    """
    from langgraph.graph import END, StateGraph

    g = StateGraph(AgentState)
    g.add_node("extract", node_extract)
    g.add_node("generate", node_generate)
    g.add_node("validate", node_validate)
    g.add_node("repair", node_repair)
    g.add_node("run", node_run)

    g.set_entry_point("extract")
    g.add_edge("extract", "generate")
    g.add_edge("generate", "validate")

    def after_validate(state: AgentState) -> str:
        if state.validation and state.validation.ok:
            if state.validate_only or not state.run_solver:
                return "done"
            return "run"
        return "repair" if state.attempts < state.settings.max_repair_attempts else "done"

    def after_run(state: AgentState) -> str:
        if state.run and state.run.ok:
            return "done"
        return "repair" if state.attempts < state.settings.max_repair_attempts else "done"

    g.add_conditional_edges(
        "validate", after_validate, {"run": "run", "repair": "repair", "done": END}
    )
    g.add_conditional_edges(
        "run", after_run, {"repair": "repair", "done": END}
    )
    g.add_edge("repair", "generate")
    return g.compile(checkpointer=checkpointer)
