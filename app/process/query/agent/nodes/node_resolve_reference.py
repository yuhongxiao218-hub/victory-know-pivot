import sys

from app.process.query.agent.state import QueryGraphState
from app.rag.query.reference_resolver_service import resolve_reference
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_resolve_reference")
def node_resolve_reference(state: QueryGraphState) -> QueryGraphState:
    """在 Planner 前解析“它/这台机器”等依赖会话历史的指代。"""
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, node_name, is_stream)

    resolution = resolve_reference(state)
    state["reference_status"] = resolution.status
    state["reference_source"] = resolution.source
    state["reference_confidence"] = resolution.confidence
    state["reference_reason"] = resolution.reason
    state["reference_candidates"] = list(resolution.item_names)

    if resolution.status == "resolved":
        state["item_names"] = list(resolution.item_names)
        state["rewritten_query"] = resolution.standalone_query

    add_done_task(session_id, node_name, is_stream)
    return state
