import sys

from app.shared.runtime.logger import node_log
from app.rag.query.tools import execute_search_tool
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_search_embedding")
def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    result = execute_search_tool("local_search", state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "embedding_chunks": result.documents,
        "tool_executions": [result.to_trace()],
    }


if __name__ == "__main__":
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "H3C LA2608怎么用？",
        "item_names": ["H3C LA2608室内无线网关"],
        "is_stream": False,
    }
    result = node_search_embedding(test_state)
    print(result)
