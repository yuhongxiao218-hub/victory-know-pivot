import sys

from app.rag.query.tools import execute_search_tool
from app.shared.runtime.logger import node_log

from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    result = execute_search_tool("hyde_search", state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "hyde_embedding_chunks": result.documents,
        "tool_executions": [result.to_trace()],
    }

if __name__ == "__main__":
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "H3C LA2608怎么用？",
        "rewritten_query": "H3C LA2608室内无线网关的具体操作步骤是什么？",
        "item_names": ["H3C LA2608室内无线网关"],
        "is_stream": False,
    }
    result = node_search_embedding_hyde(mock_state)
    print(result)
