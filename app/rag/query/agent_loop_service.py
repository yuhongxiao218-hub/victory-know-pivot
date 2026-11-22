from app.process.query.agent.state import QueryGraphState
from app.rag.query.plan_executor_service import (
    AgentExecutionLimitError,
    ToolExecutor,
    execute_next_plan_step,
)
from app.rag.query.replanner_service import ReplannerInvoker, decide_next_action
from app.rag.query.step_evaluator_service import StepEvaluatorInvoker, evaluate_latest_step
from app.rag.query.tools.registry import execute_search_tool


def run_agent_loop(
    state: QueryGraphState,
    tool_executor: ToolExecutor = execute_search_tool,
    evaluator_invoker: StepEvaluatorInvoker | None = None,
    replanner_invoker: ReplannerInvoker | None = None,
) -> QueryGraphState:
    """运行最小的执行、评估与重规划闭环，直到完成或需要澄清。"""
    state["agent_status"] = "ready"
    # shortcut: 循环在当前进程内运行，需要断点恢复时迁移到持久化执行器。
    while state["agent_status"] == "ready":
        try:
            execute_next_plan_step(state, tool_executor)
        except AgentExecutionLimitError as exc:
            state["agent_status"] = "failed"
            state["agent_finish_reason"] = str(exc)
            break

        if state.get("agent_tool_failure_count", 0) >= state.get("agent_max_tool_failures", 0):
            state["agent_status"] = "failed"
            state["agent_finish_reason"] = "工具失败预算已用完"
            break

        evaluate_latest_step(state, evaluator_invoker)
        decide_next_action(state, replanner_invoker)

    return state
