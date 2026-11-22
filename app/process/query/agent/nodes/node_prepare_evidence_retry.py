import sys

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_prepare_evidence_retry")
def node_prepare_evidence_retry(state: QueryGraphState) -> QueryGraphState:
    """把 Grader 的重试计划转换成 Router 可执行的工具列表。"""
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, node_name, is_stream)

    state["selected_tools"] = list(state.get("evidence_retry_tools", []))
    state["evidence_retry_count"] = state.get("evidence_retry_count", 0) + 1

    add_done_task(session_id, node_name, is_stream)
    return state
