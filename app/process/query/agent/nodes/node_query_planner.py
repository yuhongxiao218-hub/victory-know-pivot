import sys

from app.process.query.agent.state import QueryGraphState
from app.rag.query.planner_service import PlannerInvoker, plan_query
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_query_planner")
def node_query_planner(
    state: QueryGraphState,
    planner_invoker: PlannerInvoker | None = None,
) -> QueryGraphState:
    """调用 Planner，并将结构化决策写入查询 State。"""
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    node_name = sys._getframe().f_code.co_name

    add_running_task(session_id, node_name, is_stream)
    decision = plan_query(state, planner_invoker=planner_invoker)

    state["planner_intent"] = decision.intent
    state["planner_route"] = decision.route
    state["actual_route"] = decision.route
    state["selected_tools"] = list(decision.selected_tools)
    state["planner_confidence"] = decision.confidence
    state["planner_reason"] = decision.reason
    state["task_plan"] = decision.task_plan.model_dump() if decision.task_plan else {}

    add_done_task(session_id, node_name, is_stream)
    return state
