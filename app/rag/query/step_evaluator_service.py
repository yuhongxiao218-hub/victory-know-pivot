from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.process.query.agent.state import QueryGraphState
from app.rag.query.evidence_grader_service import format_evidence, invoke_grader_model
from app.rag.query.planner_schema import TaskPlan
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


class StepEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_completed: bool
    evidence_sufficient: bool
    reason: str = Field(min_length=1, max_length=300)


StepEvaluatorInvoker = Callable[[str], dict[str, Any]]


def evaluate_latest_step(
    state: QueryGraphState,
    evaluator_invoker: StepEvaluatorInvoker | None = None,
) -> StepEvaluation:
    """根据最新工具结果和成功标准评估当前步骤。"""
    plan = TaskPlan.model_validate(state.get("task_plan"))
    observations = state.get("step_observations", [])
    if not observations:
        raise ValueError("没有可评估的步骤执行结果")

    observation = observations[-1]
    step = next((item for item in plan.steps if item.id == observation.get("step_id")), None)
    if step is None:
        raise ValueError("步骤执行结果不属于当前任务计划")

    if not observation.get("success"):
        evaluation = StepEvaluation(
            step_completed=False,
            evidence_sufficient=False,
            reason=observation.get("error") or "工具执行失败",
        )
    elif not observation.get("documents", []):
        evaluation = StepEvaluation(
            step_completed=False,
            evidence_sufficient=False,
            reason="工具执行成功，但没有返回可用证据",
        )
    else:
        prompt = load_prompt(
            "step_evaluator",
            step_description=step.description,
            success_criteria=step.success_criteria,
            evidence=format_evidence(observation["documents"]),
        )
        invoke = evaluator_invoker or invoke_grader_model
        try:
            evaluation = StepEvaluation.model_validate(invoke(prompt))
        except Exception as exc:
            if not isinstance(exc, ValidationError):
                logger.exception("步骤评估模型调用失败")
            else:
                logger.warning(f"步骤评估结果校验失败: {exc}")
            evaluation = StepEvaluation(
                step_completed=True,
                evidence_sufficient=False,
                reason="步骤已执行，但评估结果不可用，证据充分性无法确认",
            )

    evaluation_data = evaluation.model_dump()
    observation["evaluation"] = evaluation_data
    state["step_evaluation"] = evaluation_data
    state["agent_status"] = "evaluated"
    return evaluation
