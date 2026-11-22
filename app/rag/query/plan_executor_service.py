from collections.abc import Callable

from app.process.query.agent.state import QueryGraphState, merge_tool_executions
from app.rag.query.planner_schema import PlannerTool, TaskPlan
from app.rag.query.tools.registry import execute_search_tool
from app.rag.query.tools.schema import SearchToolResult


ToolExecutor = Callable[[PlannerTool, QueryGraphState], SearchToolResult]


class AgentExecutionLimitError(RuntimeError):
    """执行预算或重复调用边界被触发。"""


def _step_call_key(tool: str, description: str) -> str:
    return f"{tool}:{' '.join(description.lower().split())}"


def execute_next_plan_step(
    state: QueryGraphState,
    tool_executor: ToolExecutor = execute_search_tool,
) -> QueryGraphState:
    """执行计划中的当前步骤，并保存工具结果供后续评估。"""
    plan = TaskPlan.model_validate(state.get("task_plan"))
    step_index = state.get("current_step_index", 0)
    if step_index >= len(plan.steps):
        raise ValueError("任务计划已没有待执行步骤")

    step = plan.steps[step_index]
    if state.get("agent_step_count", 0) >= state.get("agent_max_steps", 0):
        raise AgentExecutionLimitError("执行步骤预算已用完")
    if state.get("agent_tool_failure_count", 0) >= state.get("agent_max_tool_failures", 0):
        raise AgentExecutionLimitError("工具失败预算已用完")

    # shortcut: 计划尚无结构化参数，暂用工具名和步骤描述识别相同动作；加入参数后改用规范化参数哈希。
    call_key = _step_call_key(step.tool, step.description)
    previous_calls = sum(
        observation.get("step_call_key") == call_key
        for observation in state.get("step_observations", [])
    )
    if previous_calls > state.get("agent_max_repeated_calls", 0):
        raise AgentExecutionLimitError("相同工具动作的重复调用预算已用完")

    result = tool_executor(step.tool, state)
    # shortcut: 步骤结果直接保存在 State，结果规模增大时改为保存外部引用。
    observation = {
        "step_id": step.id,
        "step_call_key": call_key,
        "tool_name": result.tool_name,
        "success": result.success,
        "documents": result.documents,
        "duration_ms": result.duration_ms,
        "error": result.error,
    }

    state["step_observations"] = [*state.get("step_observations", []), observation]
    state["tool_executions"] = merge_tool_executions(
        state.get("tool_executions", []),
        [result.to_trace()],
    )
    state["agent_step_count"] = state.get("agent_step_count", 0) + 1
    if not result.success:
        state["agent_tool_failure_count"] = state.get("agent_tool_failure_count", 0) + 1
    state["agent_status"] = "awaiting_evaluation"
    if result.success:
        state["current_step_index"] = step_index + 1

    return state
