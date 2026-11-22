import sys

from app.process.query.agent.state import QueryGraphState
from app.rag.query.route_response_service import build_direct_response, save_routed_user_message
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_direct_answer")
def node_direct_answer(state: QueryGraphState) -> QueryGraphState:
    """处理闲聊或当前系统不支持的意图，不调用检索工具。"""
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, node_name, is_stream)
    state["answer"] = build_direct_response(state)
    save_routed_user_message(state)
    add_done_task(session_id, node_name, is_stream)
    return state
