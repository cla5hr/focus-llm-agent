"""Regression tests for user-parameter handling.

Covers the bugs where end_time / geometry / mesh values from the prompt (or
from explicit UI overrides) were silently ignored or clobbered by auto-derived
defaults.
"""

from cfd_copilot.agent import AgentState, _apply_overrides, node_extract
from cfd_copilot.llm import extract_spec
from cfd_copilot.schema import CaseSpec, CaseType


# --- Bug 1: end_time was not extractable at all -----------------------------
def test_end_time_extracted_from_prompt():
    s = extract_spec("lid-driven cavity at 1 m/s, end time 2 seconds", use_llm=False)
    assert s.control.end_time == 2.0


def test_delta_t_extracted_from_prompt():
    s = extract_spec("cavity at 1 m/s, time step 0.001", use_llm=False)
    assert s.control.delta_t == 0.001


# --- Bug 2: forward_step clobbered user end_time -----------------------------
def test_forward_step_respects_user_end_time():
    s = extract_spec("forward step at Mach 3, end time 0.005", use_llm=False)
    assert s.case_type == CaseType.FORWARD_STEP
    assert s.control.end_time == 0.005


def test_forward_step_still_derives_end_time_when_unset():
    s = extract_spec("supersonic flow over a forward step at Mach 3", use_llm=False)
    assert s.control.end_time is not None and s.control.end_time < 0.1


# --- Bug 3: rule parser never extracted geometry / mesh ----------------------
def test_height_and_length_extracted():
    s = extract_spec(
        "turbulent channel at 10 m/s, length 5 m, height 0.5 m", use_llm=False
    )
    assert s.geometry.length == 5.0
    assert s.geometry.height == 0.5


def test_forward_step_height_changes_case():
    s = extract_spec("forward step at Mach 3, height 2 m", use_llm=False)
    assert s.geometry.height == 2.0


def test_mesh_extracted():
    s = extract_spec("cavity at 1 m/s on a 100x100 mesh", use_llm=False)
    assert (s.mesh.nx, s.mesh.ny) == (100, 100)


# --- Bug 4: silent LLM fallback is now visible --------------------------------
def test_extraction_method_recorded():
    s = extract_spec("cavity at 1 m/s", use_llm=False)
    assert s.extraction_method == "rules"


# --- Bug 5: explicit 1.0 m length no longer expanded to 3.0 -------------------
def test_forward_step_explicit_unit_length_kept():
    s = extract_spec("forward step at Mach 3, length 1 m", use_llm=False)
    assert s.geometry.length == 1.0


def test_forward_step_default_length_still_3():
    s = CaseSpec(case_type=CaseType.FORWARD_STEP, mach=3.0)
    assert s.geometry.length == 3.0


# --- UI overrides (AgentState.spec_overrides) ---------------------------------
def test_spec_overrides_win_over_extraction():
    state = AgentState(
        prompt="lid-driven cavity at 1 m/s, end time 2 seconds",
        use_llm=False,
        spec_overrides={"control.end_time": 4.0, "geometry.height": 2.5},
    )
    node_extract(state)
    assert state.spec.control.end_time == 4.0
    assert state.spec.geometry.height == 2.5
    # write_interval kept consistent with the overridden end_time
    assert abs(state.spec.control.write_interval - 0.8) < 1e-9


def test_overrides_with_all_none_are_noop():
    base = extract_spec("cavity at 1 m/s", use_llm=False)
    same = _apply_overrides(base, {"control.end_time": None})
    assert same is base
