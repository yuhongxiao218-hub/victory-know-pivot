import sys

from app.process.query.agent.state import QueryGraphState
from app.rag.query.route_response_service import build_clarification_response, save_routed_user_message
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_clarify")
def node_clarify(state: QueryGraphState) -> QueryGraphState:
    """信息不足时向用户追问，不执行无依据的检索。"""
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, node_name, is_stream)
    state["actual_route"] = "clarify"
    state["answer"] = build_clarification_response(state)
    save_routed_user_message(state)
    add_done_task(session_id, node_name, is_stream)
    return state
