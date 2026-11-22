from typing import Any

from app.process.query.agent.state import QueryGraphState


def _source_summary(document: dict[str, Any]) -> dict[str, Any]:
    """只暴露前端展示需要的来源元数据，不传输大段文档正文。"""
    raw_chunk_id = document.get("chunk_id")
    return {
        "title": str(document.get("title") or document.get("name") or "未命名资料"),
        "type": str(document.get("type") or "unknown"),
        "url": str(document.get("url") or document.get("source_url") or ""),
        "chunk_id": str(raw_chunk_id) if raw_chunk_id is not None else None,
        "file_title": document.get("file_title"),
        "parent_title": document.get("parent_title"),
        "item_name": document.get("item_name"),
        "part": document.get("part"),
    }


def _source_key(document: dict[str, Any], source: dict[str, Any]) -> tuple[Any, ...]:
    """本地片段优先按真实 chunk_id 去重，其余来源回退到标题和 URL。"""
    chunk_id = document.get("chunk_id")
    if source.get("type") != "web" and chunk_id is not None:
        return ("chunk", source.get("type"), chunk_id)
    return ("title_url", source.get("title"), source.get("url"))


def build_agent_trace(
    state: QueryGraphState,
    *,
    include_contexts: bool = False,
) -> dict[str, Any]:
    """把 Graph 最终状态转换为稳定、轻量、适合前端展示的 Agent 摘要。"""
    raw_plan = state.get("task_plan") or {}
    plan_steps = raw_plan.get("steps", []) if isinstance(raw_plan, dict) else []
    plan = {
        "goal": str(raw_plan.get("goal") or "") if isinstance(raw_plan, dict) else "",
        "steps": [
            {
                "id": str(step.get("id") or ""),
                "description": str(step.get("description") or ""),
                "tool": str(step.get("tool") or ""),
                "success_criteria": str(step.get("success_criteria") or ""),
            }
            for step in plan_steps
            if isinstance(step, dict)
        ],
    }

    observations = []
    for observation in state.get("step_observations", []):
        documents = observation.get("documents", [])
        observations.append(
            {
                "step_id": str(observation.get("step_id") or ""),
                "tool": str(observation.get("tool_name") or "unknown"),
                "success": bool(observation.get("success")),
                "result_count": len(documents),
                "results": [_source_summary(document) for document in documents[:3]],
                "duration_ms": int(observation.get("duration_ms") or 0),
                "error": str(observation.get("error") or ""),
                "evaluation": dict(observation.get("evaluation") or {}),
            }
        )

    tool_executions = [
        {
            "name": str(record.get("tool_name") or "unknown"),
            "success": bool(record.get("success")),
            "result_count": int(record.get("result_count") or 0),
            "duration_ms": int(record.get("duration_ms") or 0),
            "error": str(record.get("error") or ""),
        }
        for record in state.get("tool_executions", [])
    ]

    sources: list[dict[str, Any]] = []
    seen_sources: set[tuple[Any, ...]] = set()
    for document in state.get("reranked_docs", []):
        source = _source_summary(document)
        source_key = _source_key(document, source)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        sources.append(source)

    trace = {
        "reference": {
            "status": state.get("reference_status", ""),
            "source": state.get("reference_source", "none"),
            "confidence": float(state.get("reference_confidence", 0.0) or 0.0),
            "reason": state.get("reference_reason", ""),
            "candidates": list(state.get("reference_candidates", [])),
        },
        "intent": state.get("planner_intent", ""),
        "route": state.get("actual_route") or state.get("planner_route", ""),
        "planner_reason": state.get("planner_reason", ""),
        "planner_confidence": float(state.get("planner_confidence", 0.0) or 0.0),
        "plan": plan,
        "steps": observations,
        "tools": tool_executions,
        "agent_status": state.get("agent_status", "idle"),
        "finish_reason": state.get("agent_finish_reason", ""),
        "step_count": int(state.get("agent_step_count", 0) or 0),
        "replan_count": int(state.get("agent_replan_count", 0) or 0),
        "tool_failure_count": int(state.get("agent_tool_failure_count", 0) or 0),
        "retry_count": int(state.get("evidence_retry_count", 0) or 0),
        "evidence": {
            "action": state.get("evidence_action", ""),
            "confidence": float(state.get("evidence_confidence", 0.0) or 0.0),
            "reason": state.get("evidence_reason", ""),
        },
        "sources": sources,
    }
    if include_contexts:
        trace["retrieved_contexts"] = [
            text.strip()
            for document in state.get("reranked_docs", [])
            if isinstance(text := document.get("text") or document.get("content"), str)
            and text.strip()
        ]
    return trace
