import json
import sys

from app.process.query.agent.state import QueryGraphState
from app.rag.query.agent_loop_service import run_agent_loop
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


def _collect_agent_evidence(state: QueryGraphState) -> list[dict]:
    """把工具观察结果转换为现有回答生成节点使用的文档格式。"""
    evidence = []
    for observation in state.get("step_observations", []):
        if not observation.get("success"):
            continue
        for document in observation.get("documents", []):
            text = document.get("text") or document.get("content") or document.get("snippet")
            if not text and document.get("comparison"):
                text = json.dumps(document["comparison"], ensure_ascii=False)
            if not text:
                continue
            evidence.append(
                {
                    **document,
                    "title": document.get("title") or document.get("name") or "Agent工具结果",
                    "text": text,
                    "type": document.get("type") or "agent",
                    "score": float(document.get("score") or 1.0),
                }
            )
    return evidence


@node_log("node_agent_execute")
def node_agent_execute(state: QueryGraphState) -> QueryGraphState:
    """执行多步 Agent 计划，并将最终证据交给现有答案生成节点。"""
    node_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, node_name, is_stream)

    run_agent_loop(state)
    state["actual_route"] = "agent"
    if state.get("agent_status") == "completed":
        state["reranked_docs"] = _collect_agent_evidence(state)
        if not state["reranked_docs"]:
            state["agent_status"] = "failed"
            state["agent_finish_reason"] = "Agent完成计划但没有产生可用于回答的证据"

    if state.get("agent_status") != "completed":
        reason = state.get("agent_finish_reason") or "Agent未能完成当前任务"
        if state.get("agent_status") == "needs_clarification":
            state["answer"] = reason
        else:
            state["answer"] = f"本次任务未能可靠完成：{reason}"

    add_done_task(session_id, node_name, is_stream)
    return state
