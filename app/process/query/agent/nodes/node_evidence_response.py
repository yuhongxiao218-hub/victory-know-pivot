import sys

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_evidence_response")
def node_evidence_response(state: QueryGraphState) -> QueryGraphState:
    """将证据不足或工具失败转换为用户可理解的答案。"""
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, node_name, is_stream)

    if state.get("evidence_action") == "fail":
        state["answer"] = "检索服务暂时不可用，当前无法可靠回答这个问题，请稍后重试。"
    elif state.get("evidence_action") == "refuse":
        state["answer"] = "我已检索本地知识库和可用的网络信息，但现有资料仍不足以可靠回答。为避免编造，我暂时无法给出结论。"
    else:
        state["answer"] = "当前检索到的资料不足以可靠回答。请补充更具体的设备型号、功能或故障现象。"

    add_done_task(session_id, node_name, is_stream)
    return state
