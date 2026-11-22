import sys

from app.process.query.agent.state import QueryGraphState
from app.rag.query.evidence_grader_service import GraderInvoker, grade_evidence
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_evidence_grader")
def node_evidence_grader(
    state: QueryGraphState,
    grader_invoker: GraderInvoker | None = None,
) -> QueryGraphState:
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, node_name, is_stream)

    grade = grade_evidence(state, grader_invoker=grader_invoker)
    state["evidence_action"] = grade.action
    state["evidence_confidence"] = grade.confidence
    state["evidence_reason"] = grade.reason
    state["evidence_retry_tools"] = list(grade.retry_tools)

    add_done_task(session_id, node_name, is_stream)
    return state
