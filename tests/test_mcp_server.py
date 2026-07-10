"""Tests for the FOCUS MCP server."""

import pytest

mcp_sdk = pytest.importorskip("mcp", reason="mcp extra not installed")

from cfd_copilot.mcp_server import (  # noqa: E402
    create_case,
    get_case_log,
    mcp,
    solve_case,
)
from cfd_copilot.openfoam import openfoam_available  # noqa: E402


@pytest.mark.anyio
async def test_all_tools_registered():
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {"create_case", "solve_case", "get_case_log"}


def test_solve_case_requires_existing_case(tmp_path):
    out = solve_case(str(tmp_path / "nope"))
    assert "error" in out


def test_get_case_log_lists_available(tmp_path):
    (tmp_path / "log.blockMesh").write_text("End\n")
    out = get_case_log(str(tmp_path))
    assert out["available_logs"] == ["log.blockMesh"]


@pytest.mark.skipif(not openfoam_available(), reason="OpenFOAM not installed")
def test_create_and_solve_over_mcp_functions(tmp_path):
    created = create_case(
        "cavity", velocity=1.0, nu=0.01, end_time=0.1, nx=20, ny=20,
        out_dir=str(tmp_path),
    )
    assert created["mesh_ok"], created["errors"]
    assert created["spec"]["control"]["end_time"] == 0.1
    solved = solve_case(created["case_dir"])
    assert solved["completed"], solved["errors"]
