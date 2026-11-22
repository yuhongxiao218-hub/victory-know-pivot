from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger


def route_after_reference_resolution(state: QueryGraphState) -> str:
    """缺失或多义指代直接澄清，其余问题再交给 Planner。"""
    status = state.get("reference_status", "not_required")
    if status in {"ambiguous", "missing"}:
        return "node_clarify"
    return "node_query_planner"


PLANNER_ROUTE_TO_NODE = {
    "direct_answer": "node_direct_answer",
    "clarify": "node_clarify",
    "device_rag": "node_item_name_confirm",
    "web_rag": "node_web_search_mcp",
    "hybrid_rag": "node_item_name_confirm",
}

TOOL_TO_NODE = {
    "local_search": "node_search_embedding",
    "hyde_search": "node_search_embedding_hyde",
    "web_search": "node_web_search_mcp",
}


def route_after_planner(state: QueryGraphState) -> str:
    """将 Planner 的结构化主路径映射为下一个图节点。"""
    plan_steps = state.get("task_plan", {}).get("steps", [])
    if plan_steps:
        needs_item_name = any(step.get("tool") in {"local_search", "hyde_search"} for step in plan_steps)
        target = "node_item_name_confirm" if needs_item_name else "node_agent_execute"
        logger.info(f"Router多步Agent路由: target={target}")
        return target

    route = state.get("planner_route", "device_rag")
    target = PLANNER_ROUTE_TO_NODE.get(route, "node_item_name_confirm")
    logger.info(f"Router主路由: route={route}, target={target}")
    return target


def route_after_item_confirmation(state: QueryGraphState) -> str | tuple[str, ...]:
    """主体确认后，根据工具列表启动一个或多个检索分支。"""
    if not state.get("item_names"):
        logger.info("未确认知识库主体，使用已有澄清答案提前结束")
        return "node_answer_output"

    if state.get("task_plan", {}).get("steps"):
        return "node_agent_execute"

    selected_tools = state.get("selected_tools") or ["local_search"]
    targets = tuple(TOOL_TO_NODE[tool] for tool in selected_tools if tool in TOOL_TO_NODE)
    if not targets:
        logger.warning("Planner未提供有效检索工具，降级到本地检索")
        return "node_search_embedding"

    logger.info(f"Router工具路由: tools={selected_tools}, targets={targets}")
    return targets[0] if len(targets) == 1 else targets


def route_after_web_search(state: QueryGraphState) -> str:
    """纯 Web 直接精排；混合检索等待本地分支在 RRF 节点汇合。"""
    if state.get("planner_route") == "web_rag":
        return "node_rerank"
    return "node_rrf"


def route_after_evidence_grader(state: QueryGraphState) -> str:
    """根据证据评估动作选择回答、重试或返回提示。"""
    action_to_node = {
        "answer": "node_answer_output",
        "retry": "node_prepare_evidence_retry",
        "clarify": "node_evidence_response",
        "refuse": "node_evidence_response",
        "fail": "node_evidence_response",
    }
    action = state.get("evidence_action", "clarify")
    target = action_to_node.get(action, "node_evidence_response")
    logger.info(f"Evidence Router: action={action}, target={target}")
    return target


def route_evidence_retry_tools(state: QueryGraphState) -> str | tuple[str, ...]:
    """执行 Grader 选择的补充工具；工具自身负责检查前置条件。"""
    targets = tuple(
        TOOL_TO_NODE[tool]
        for tool in state.get("selected_tools", [])
        if tool in TOOL_TO_NODE
    )
    if not targets:
        return "node_evidence_response"
    return targets[0] if len(targets) == 1 else targets
