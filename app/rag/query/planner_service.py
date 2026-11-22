import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.planner_schema import PlannerDecision, default_planner_decision
from app.shared.model.lm_utils import require_json_object
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log


PlannerInvoker = Callable[[str], dict[str, Any]]


def format_planner_history(history: list[dict], limit: int = 10) -> str:
    """将最近会话压缩为 Planner 可读取的 JSON，避免传入数据库内部字段。"""
    if not history:
        return "[]"

    compact_history = [
        {
            "role": message.get("role", ""),
            "text": message.get("text", ""),
            "rewritten_query": message.get("rewritten_query", ""),
            "item_names": message.get("item_names") or [],
        }
        for message in history[-limit:]
    ]
    return json.dumps(compact_history, ensure_ascii=False)


def invoke_planner_model(prompt: str) -> dict[str, Any]:
    """调用支持 JSON 模式的文本模型，返回 Planner 原始字典结果。"""
    llm_client = llm_provider.chat(json_model=True)
    chain = llm_client | JsonOutputParser()
    return chain.invoke([HumanMessage(content=prompt)])


@step_log("plan_query")
def plan_query(
    state: QueryGraphState,
    planner_invoker: PlannerInvoker | None = None,
) -> PlannerDecision:
    """根据用户问题和会话历史生成经过 Pydantic 校验的规划决策。"""
    question = (state.get("rewritten_query") or state.get("original_query") or "").strip()
    if not question:
        return PlannerDecision(
            intent="ambiguous",
            route="clarify",
            selected_tools=[],
            confidence=1.0,
            reason="用户问题为空，需要用户补充问题内容",
        )

    prompt = load_prompt(
        "query_planner",
        question=question,
        history=format_planner_history(state.get("history", [])),
    )
    invoke = planner_invoker or invoke_planner_model

    #Planner模型调用失败
    try:
        """
        raw_decision
        = LLM交上来的答卷
        = 尚未批改

        decision
        = Pydantic检查后的答卷
        = 已经通过格式和逻辑校验

        两个try为了区分
        模型没成功返回结果
        和：模型返回了结果，但结果不符合系统要求

        """
        raw_decision = require_json_object(invoke(prompt))
    except Exception:
        logger.exception("Planner模型调用失败，降级到原有本地RAG流程")
        return default_planner_decision("Planner调用失败，降级到原有本地RAG流程")
    #Planner结构化结果校验失败
    try:
        decision = PlannerDecision.model_validate(raw_decision)
    except ValidationError as exc:
        logger.warning(f"Planner结构化结果校验失败，执行降级: {exc}")
        return default_planner_decision("Planner输出无效，降级到原有本地RAG流程")

    logger.info(
        "Planner规划完成: "
        f"intent={decision.intent}, route={decision.route}, "
        f"tools={decision.selected_tools}, confidence={decision.confidence}"
    )
    return decision
