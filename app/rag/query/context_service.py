from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import step_log


@step_log("load_query_context")
def load_query_context(state: QueryGraphState, limit: int = 20) -> list[dict]:
    """在 Planner 执行前加载当前会话的近期历史。"""
    session_id = state.get("session_id", "").strip()
    if not session_id:
        raise ValueError("加载会话历史需要 session_id")
    return history_repository.list_recent(session_id, limit=limit)
