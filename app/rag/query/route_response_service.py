from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import step_log


@step_log("build_direct_response")
def build_direct_response(state: QueryGraphState) -> str:
    """为无需检索的问题构造确定性回答。"""
    if state.get("planner_intent") == "chat":
        return "你好！我是知枢智能助手，可以帮你查询设备操作、配置、参数和故障资料。"
    return "当前我主要支持设备知识查询和相关信息检索，请换一种方式描述你的问题。"


@step_log("build_clarification_response")
def build_clarification_response(state: QueryGraphState) -> str:
    """当信息不足时请求用户补充，而不是盲目检索。"""
    candidates = state.get("reference_candidates", [])
    if state.get("reference_status") == "ambiguous" and candidates:
        options = "、".join(f"“{name}”" for name in candidates)
        return f"你说的设备可能是{options}。请告诉我具体指的是哪一个。"
    if state.get("reference_status") == "missing":
        return "我还不知道“这台机器”具体指哪一个设备。请先补充设备名称或型号。"
    return "我还无法确定你咨询的具体设备。请补充设备名称或型号，例如：H3C LA2608 怎么配置？"


@step_log("save_routed_user_message")
def save_routed_user_message(state: QueryGraphState) -> None:
    """保存未进入主体确认流程的用户消息，保证多轮历史完整。"""
    original_query = state.get("original_query", "")
    state["rewritten_query"] = state.get("rewritten_query") or original_query
    history_repository.save_message(
        session_id=state["session_id"],
        role="user",
        text=original_query,
        rewritten_query=state["rewritten_query"],
        item_names=state.get("item_names", []),
        image_urls=[],
    )
