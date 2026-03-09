__all__ = [
    "build_dashboard_model",
    "list_runs",
    "render_run_dashboard_html",
    "resolve_run_id",
    "write_run_dashboard",
]


def __getattr__(name: str):
    if name in {
        "build_dashboard_model",
        "list_runs",
        "render_run_dashboard_html",
        "resolve_run_id",
        "write_run_dashboard",
    }:
        from .agent_empire import (
            build_dashboard_model,
            list_runs,
            render_run_dashboard_html,
            resolve_run_id,
            write_run_dashboard,
        )

        return {
            "build_dashboard_model": build_dashboard_model,
            "list_runs": list_runs,
            "render_run_dashboard_html": render_run_dashboard_html,
            "resolve_run_id": resolve_run_id,
            "write_run_dashboard": write_run_dashboard,
        }[name]
    raise AttributeError(name)
