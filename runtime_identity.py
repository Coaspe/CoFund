"""
runtime_identity.py
===================
Shared runtime identity helpers.

- agent: responsibility owner in the investment team; may own one or more graph nodes
- owner agent: responsibility owner for a node, even when the node itself is system-owned
- node: concrete LangGraph execution step
"""

from __future__ import annotations

from typing import Any


_AGENT_ID_BY_NODE_NAME: dict[str, str] = {
    # Legacy desk event names
    "macro": "macro",
    "fundamental": "fundamental",
    "sentiment": "sentiment",
    "quant": "quant",
    # Current graph node names
    "orchestrator": "orchestrator",
    "research_manager": "research_manager",
    "macro_analyst": "macro",
    "macro_analyst_research": "macro",
    "fundamental_analyst": "fundamental",
    "fundamental_analyst_research": "fundamental",
    "sentiment_analyst": "sentiment",
    "sentiment_analyst_research": "sentiment",
    "quant_analyst": "quant",
    "quant_analyst_research": "quant",
    "monitoring_router": "research_manager",
    "research_router": "research_manager",
    "research_executor": "research_manager",
    "research_barrier": "research_manager",
    "risk_manager": "risk_manager",
    "report_writer": "report_writer",
}

_OWNER_AGENT_ID_BY_NODE_NAME: dict[str, str] = {
    **_AGENT_ID_BY_NODE_NAME,
}


def event_node_name(event: dict[str, Any]) -> str:
    return str(event.get("node_name", "")).strip()


def agent_id_for_node(node_name: str) -> str | None:
    key = str(node_name or "").strip()
    if not key:
        return None
    return _AGENT_ID_BY_NODE_NAME.get(key)


def owner_agent_id_for_node(node_name: str) -> str | None:
    key = str(node_name or "").strip()
    if not key:
        return None
    return _OWNER_AGENT_ID_BY_NODE_NAME.get(key)


def event_agent_id(event: dict[str, Any]) -> str | None:
    raw = event.get("agent_id")
    explicit = raw.strip() if isinstance(raw, str) else ""
    if explicit:
        return explicit
    return agent_id_for_node(event_node_name(event))


def event_owner_agent_id(event: dict[str, Any]) -> str | None:
    raw = event.get("owner_agent_id")
    explicit = raw.strip() if isinstance(raw, str) else ""
    if explicit:
        return explicit
    return owner_agent_id_for_node(event_node_name(event))


def dashboard_node_id_for_event(event: dict[str, Any]) -> str:
    agent_id = event_agent_id(event)
    if agent_id:
        return agent_id
    node_name = event_node_name(event)
    return node_name or "unknown"
