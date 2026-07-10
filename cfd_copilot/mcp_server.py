"""FOCUS MCP server.

Exposes FOCUS (FOam Copilot for User-driven Simulation) as a set of MCP tools
so that any MCP host (Claude Desktop, Cursor, ...) can set up, validate, and
run OpenFOAM cases through natural conversation.

Design note: in the standalone app, a local Ollama model extracts the CaseSpec
from natural language. Over MCP, the *host* LLM plays that role -- it fills the
structured tool arguments directly. The server therefore exposes primitives
(create -> validate -> solve -> inspect logs) rather than one monolithic
"do everything" tool, which lets the host model act as the repair agent itself:
read the distilled solver error, adjust a parameter, and retry.

Run with:  focus-mcp            (stdio transport, for Claude Desktop / Cursor)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP

from cfd_copilot.generator import generate_case
from cfd_copilot.runner import run_solver
from cfd_copilot.schema import (
    CaseSpec,
    CaseType,
    ControlSpec,
    FluidProperties,
    GeometrySpec,
    MeshSpec,
    TurbulenceModel,
)
from cfd_copilot.validator import extract_foam_errors, validate_case

mcp = FastMCP(
    "focus",
    instructions=(
        "FOCUS turns flow-problem descriptions into validated, runnable OpenFOAM "
        "cases. Workflow: create_case (mesh is auto-validated) -> solve_case -> "
        "get_case_log to debug failures. Case types: lid-driven cavity (icoFoam), "
        "channel flow (simpleFoam), forward step / supersonic (rhoCentralFoam). "
        "If a solve diverges, recreate the case with a smaller delta_t."
    ),
)

_DEFAULT_OUT = Path("runs")


def _spec_summary(spec: CaseSpec) -> dict:
    return {
        "name": spec.name,
        "case_type": spec.case_type.value,
        "solver": spec.solver.value,
        "velocity_m_s": spec.velocity,
        "mach": spec.mach,
        "turbulence": spec.turbulence.value,
        "reynolds": round(spec.reynolds(), 3),
        "geometry": {"length": spec.geometry.length, "height": spec.geometry.height},
        "mesh": {"nx": spec.mesh.nx, "ny": spec.mesh.ny},
        "control": {
            "end_time": spec.control.end_time,
            "delta_t": spec.control.delta_t,
            "max_iterations": spec.control.max_iterations,
        },
    }


@mcp.tool()
def create_case(
    case_type: Literal["cavity", "channel", "forward_step"],
    name: Optional[str] = None,
    velocity: Optional[float] = None,
    mach: Optional[float] = None,
    nu: Optional[float] = None,
    temperature: Optional[float] = None,
    pressure: Optional[float] = None,
    turbulence: Optional[Literal["laminar", "kEpsilon", "kOmegaSST"]] = None,
    length: Optional[float] = None,
    height: Optional[float] = None,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    end_time: Optional[float] = None,
    delta_t: Optional[float] = None,
    out_dir: str = "runs",
) -> dict:
    """Create an OpenFOAM case from structured parameters and validate its mesh.

    Every parameter except case_type is optional -- omitted values get sensible,
    physics-aware defaults (e.g. for a forward step, Mach + temperature derive
    the velocity, and end_time defaults to ~4 flow-through times). Returns the
    resolved spec, the case directory, and the blockMesh/checkMesh validation
    report. The case is ready to solve only if `mesh_ok` is true.
    """
    fluid = FluidProperties()
    if nu is not None:
        fluid.nu = nu
    if temperature is not None:
        fluid.T = temperature
    if pressure is not None:
        fluid.p = pressure

    geometry = GeometrySpec()
    if length is not None:
        geometry.length = length
    if height is not None:
        geometry.height = height

    mesh = MeshSpec()
    if nx is not None:
        mesh.nx = nx
    if ny is not None:
        mesh.ny = ny

    control = ControlSpec()
    if end_time is not None:
        control.end_time = end_time
    if delta_t is not None:
        control.delta_t = delta_t

    kwargs = dict(
        case_type=CaseType(case_type),
        fluid=fluid,
        geometry=geometry,
        mesh=mesh,
        control=control,
        extraction_method="mcp-host",
    )
    kwargs["name"] = name or f"{case_type}_mcp"
    if velocity is not None:
        kwargs["velocity"] = velocity
    if mach is not None:
        kwargs["mach"] = mach
    if turbulence is not None:
        kwargs["turbulence"] = TurbulenceModel(turbulence)

    spec = CaseSpec(**kwargs)
    case_dir = generate_case(spec, Path(out_dir))

    report = validate_case(case_dir)
    return {
        "case_dir": str(case_dir),
        "spec": _spec_summary(spec),
        "mesh_ok": report.ok,
        "mesh": {
            "n_cells": report.n_cells,
            "max_non_orthogonality": report.max_non_ortho,
            "max_skewness": report.max_skewness,
        },
        "errors": report.errors,
        "warnings": report.warnings,
        "next_step": "call solve_case with this case_dir" if report.ok else
                     "mesh invalid -- adjust mesh/geometry parameters and recreate",
    }


@mcp.tool()
def solve_case(case_dir: str, timeout_s: int = 1800) -> dict:
    """Run the appropriate OpenFOAM solver on a previously created case.

    The solver (icoFoam / simpleFoam / rhoCentralFoam) is read from the
    case_spec.json stored in the case directory. Returns completion status,
    convergence info, and final residuals per field. If `completed` is false,
    call get_case_log for the distilled error, then typically recreate the case
    with a smaller delta_t (transient divergence) or more iterations (steady).
    """
    path = Path(case_dir)
    spec_file = path / "case_spec.json"
    if not spec_file.is_file():
        return {"error": f"no case_spec.json found in {case_dir}; create the case first"}
    spec = CaseSpec(**json.loads(spec_file.read_text()))

    report = run_solver(spec, path, timeout=timeout_s)
    return {
        "case_dir": str(path),
        "solver": spec.solver.value,
        "completed": report.completed,
        "converged": report.converged,
        "last_time_or_iteration": report.last_time,
        "final_residuals": report.final_residuals,
        "summary": report.summary(),
        "errors": report.errors,
    }


@mcp.tool()
def get_case_log(case_dir: str, log_name: Optional[str] = None, tail_lines: int = 40) -> dict:
    """Inspect a case's log files for diagnosis.

    Returns the distilled OpenFOAM FATAL errors plus the last `tail_lines` of
    the requested log (e.g. 'log.blockMesh', 'log.icoFoam'). If log_name is
    omitted, lists the available logs.
    """
    path = Path(case_dir)
    logs = sorted(p.name for p in path.glob("log.*"))
    if log_name is None:
        return {"available_logs": logs}
    log_path = path / log_name
    if not log_path.is_file():
        return {"error": f"{log_name} not found", "available_logs": logs}
    text = log_path.read_text(errors="replace")
    return {
        "log": log_name,
        "distilled_errors": extract_foam_errors(text),
        "tail": "\n".join(text.splitlines()[-tail_lines:]),
    }


def main() -> None:
    """Entry point for the `focus-mcp` console script (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
