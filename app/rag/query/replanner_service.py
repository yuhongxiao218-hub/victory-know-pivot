import json
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.process.query.agent.state import QueryGraphState
from app.rag.query.evidence_grader_service import invoke_grader_model
from app.rag.query.planner_schema import TaskPlan
from app.rag.query.step_evaluator_service import StepEvaluation
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


NextAction = Literal["continue", "replan", "clarify", "finish"]
ReplannerInvoker = Callable[[str], dict[str, Any]]


class ReplanDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: NextAction
    reason: str = Field(min_length=1, max_length=300)
    revised_plan: TaskPlan | None = None
    clarification: str | None = Field(default=None, min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_action_payload(self) -> "ReplanDecision":
        if self.action == "replan" and self.revised_plan is None:
            raise ValueError("replan 动作必须提供 revised_plan")
        if self.action == "clarify" and not self.clarification:
            raise ValueError("clarify 动作必须提供 clarification")
        if self.action != "replan" and self.revised_plan is not None:
            raise ValueError(f"{self.action} 动作不应提供 revised_plan")
        if self.action != "clarify" and self.clarification is not None:
            raise ValueError(f"{self.action} 动作不应提供 clarification")
        return self


def decide_next_action(
    state: QueryGraphState,
    replanner_invoker: ReplannerInvoker | None = None,
) -> ReplanDecision:
    """根据步骤评估结果决定继续、重规划、澄清或结束。"""
    plan = TaskPlan.model_validate(state.get("task_plan"))
    evaluation = StepEvaluation.model_validate(state.get("step_evaluation"))

    if evaluation.step_completed and evaluation.evidence_sufficient:
        if state.get("current_step_index", 0) < len(plan.steps):
            decision = ReplanDecision(action="continue", reason="当前步骤完成，继续执行剩余计划")
        else:
            decision = ReplanDecision(action="finish", reason="全部计划步骤均已完成且证据充分")
        return apply_replan_decision(state, decision)

    if state.get("agent_replan_count", 0) >= state.get("agent_max_replans", 0):
        decision = ReplanDecision(
            action="clarify",
            reason="重规划预算已用完",
            clarification="现有证据不足，请补充设备型号或更具体的任务条件。",
        )
        return apply_replan_decision(state, decision)

    prompt = load_prompt(
        "replanner",
        plan=json.dumps(plan.model_dump(), ensure_ascii=False),
        evaluation=json.dumps(evaluation.model_dump(), ensure_ascii=False),
        replan_budget=state.get("agent_max_replans", 0) - state.get("agent_replan_count", 0),
    )
    invoke = replanner_invoker or invoke_grader_model
    try:
        decision = ReplanDecision.model_validate(invoke(prompt))
    except Exception as exc:
        if not isinstance(exc, ValidationError):
            logger.exception("重规划模型调用失败")
        else:
            logger.warning(f"重规划结果校验失败: {exc}")
        decision = ReplanDecision(
            action="clarify",
            reason="无法生成可靠的后续计划",
            clarification="现有证据不足，请补充设备型号或更具体的任务条件。",
        )
    return apply_replan_decision(state, decision)


def apply_replan_decision(state: QueryGraphState, decision: ReplanDecision) -> ReplanDecision:
    if decision.action == "continue":
        state["agent_status"] = "ready"
        state["step_evaluation"] = {}
    elif decision.action == "replan":
        state["task_plan"] = decision.revised_plan.model_dump()
        state["current_step_index"] = 0
        state["agent_replan_count"] = state.get("agent_replan_count", 0) + 1
        state["agent_status"] = "ready"
        state["step_evaluation"] = {}
    elif decision.action == "clarify":
        state["agent_status"] = "needs_clarification"
        state["agent_finish_reason"] = decision.clarification or decision.reason
    else:
        evaluation = StepEvaluation.model_validate(state.get("step_evaluation"))
        state["agent_status"] = "completed" if evaluation.evidence_sufficient else "failed"
        state["agent_finish_reason"] = decision.reason
    return decision
