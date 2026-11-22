import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.evidence_grader_schema import EvidenceGrade
from app.rag.query.planner_schema import PlannerTool
from app.shared.model.lm_utils import require_json_object
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log


MAX_EVIDENCE_RETRIES = 1
EVIDENCE_FALLBACK_SCORE = 0.35
GraderInvoker = Callable[[str], dict[str, Any]]


def invoke_grader_model(prompt: str) -> dict[str, Any]:
    llm_client = llm_provider.chat(json_model=True)
    chain = llm_client | JsonOutputParser()
    return require_json_object(chain.invoke([HumanMessage(content=prompt)]))


def format_evidence(documents: list[dict], limit: int = 5, text_limit: int = 1200) -> str:
    """压缩证据，避免把完整长文档全部交给 Grader。"""
    evidence = [
        {
            "index": index,
            "title": document.get("title", ""),
            "type": document.get("type", ""),
            "score": document.get("score", 0.0),
            "text": (
                document.get("text")
                or document.get("content")
                or document.get("snippet")
                or ""
            )[:text_limit],
        }
        for index, document in enumerate(documents[:limit], start=1)
    ]
    return json.dumps(evidence, ensure_ascii=False)


def eligible_retry_tools(state: QueryGraphState) -> list[PlannerTool]:
    """只返回未尝试且满足当前前置条件的工具。"""
    attempted = {
        record.get("tool_name")
        for record in state.get("tool_executions", [])
        if record.get("tool_name")
    }
    # 阶段4策略：本地证据不足时优先联网兜底，再考虑HyDE等其它工具。
    candidates: list[PlannerTool] = ["web_search", "hyde_search"]
    if state.get("item_names"):
        candidates.append("local_search")
    else:
        candidates = ["web_search"]
    return [tool for tool in candidates if tool not in attempted]


def apply_retry_policy(
    state: QueryGraphState,
    requested_tools: list[PlannerTool],
) -> list[PlannerTool]:
    """约束模型重试建议：本地检索后尚未联网时，强制优先 Web 兜底。"""
    available_tools = eligible_retry_tools(state)
    attempted_tools = {
        record.get("tool_name")
        for record in state.get("tool_executions", [])
        if record.get("tool_name")
    }
    local_was_attempted = bool(attempted_tools.intersection({"local_search", "hyde_search"}))
    if local_was_attempted and "web_search" in available_tools:
        return ["web_search"]

    allowed = set(available_tools)
    selected = [tool for tool in requested_tools if tool in allowed]
    if selected:
        return selected
    return available_tools[:1]


def grade_without_evidence(state: QueryGraphState) -> EvidenceGrade:
    """没有候选证据时使用确定性规则，不浪费一次 LLM 调用。"""
    executions = state.get("tool_executions", [])
    retry_count = state.get("evidence_retry_count", 0)
    retry_candidates = eligible_retry_tools(state)

    if retry_count < MAX_EVIDENCE_RETRIES and retry_candidates:
        return EvidenceGrade(
            action="retry",
            confidence=1.0,
            reason="当前没有可用证据，尝试尚未执行的替代检索工具",
            retry_tools=[retry_candidates[0]],
        )

    if executions and all(not record.get("success", False) for record in executions):
        return EvidenceGrade(
            action="fail",
            confidence=1.0,
            reason="所有已执行的检索工具均失败",
        )

    return EvidenceGrade(
        action="refuse",
        confidence=1.0,
        reason="本地与可用补充检索均已完成，但没有找到足以回答问题的证据",
    )


def fallback_grade(state: QueryGraphState, reason: str) -> EvidenceGrade:
    """Grader 模型不可用时，根据最高重排分数保守降级。"""
    documents = state.get("reranked_docs", [])
    top_score = max((doc.get("score", 0.0) for doc in documents), default=0.0)
    if top_score >= EVIDENCE_FALLBACK_SCORE:
        return EvidenceGrade(
            action="answer",
            confidence=0.0,
            reason=f"{reason}；最高重排分数达到降级阈值",
        )

    retry_candidates = eligible_retry_tools(state)
    if state.get("evidence_retry_count", 0) < MAX_EVIDENCE_RETRIES and retry_candidates:
        return EvidenceGrade(
            action="retry",
            confidence=0.0,
            reason=f"{reason}；最高重排分数较低，尝试替代工具",
            retry_tools=[retry_candidates[0]],
        )
    return EvidenceGrade(
        action="refuse",
        confidence=0.0,
        reason=f"{reason}；证据相关性不足",
    )


@step_log("grade_evidence")
def grade_evidence(
    state: QueryGraphState,
    grader_invoker: GraderInvoker | None = None,
) -> EvidenceGrade:
    """评估重排后的证据是否足以支持回答，并给出下一步动作。"""
    documents = state.get("reranked_docs", [])
    if not documents:
        return grade_without_evidence(state)

    prompt = load_prompt(
        "evidence_grader",
        question=state.get("rewritten_query") or state.get("original_query") or "",
        item_names=json.dumps(state.get("item_names", []), ensure_ascii=False),
        tool_executions=json.dumps(state.get("tool_executions", []), ensure_ascii=False),
        evidence=format_evidence(documents),
    )
    invoke = grader_invoker or invoke_grader_model

    try:
        raw_grade = invoke(prompt)
    except Exception:
        logger.exception("Evidence Grader 模型调用失败，使用重排分数降级")
        return fallback_grade(state, "Grader调用失败")

    try:
        grade = EvidenceGrade.model_validate(raw_grade)
    except ValidationError as exc:
        logger.warning(f"Evidence Grader 结构校验失败，执行降级: {exc}")
        return fallback_grade(state, "Grader输出无效")

    if grade.action == "retry":
        retry_tools = apply_retry_policy(state, grade.retry_tools)
        if state.get("evidence_retry_count", 0) >= MAX_EVIDENCE_RETRIES or not retry_tools:
            return EvidenceGrade(
                action="refuse",
                confidence=grade.confidence,
                reason="补充检索机会已用完，现有证据仍不足以可靠回答",
            )
        grade.retry_tools = retry_tools

    logger.info(
        f"证据评估完成: action={grade.action}, confidence={grade.confidence}, "
        f"retry_tools={grade.retry_tools}"
    )
    return grade
