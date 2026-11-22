import sys

from app.process.query.agent.state import QueryGraphState
from app.rag.query.context_service import load_query_context
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_load_context")
def node_load_context(state: QueryGraphState) -> QueryGraphState:
    """在规划前加载历史上下文。"""
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, node_name, is_stream)
    state["history"] = load_query_context(state)
    add_done_task(session_id, node_name, is_stream)
    return state
