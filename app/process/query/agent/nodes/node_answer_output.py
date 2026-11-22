import sys

from app.shared.runtime.logger import node_log
from app.rag.query.answer_service import generate_answer
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_answer_output")
def node_answer_output(state):
    """
    节点功能：生成最终回答并交付给用户（支持流式/非流式）。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    state = generate_answer(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state["is_stream"])
    return state

if __name__ == "__main__":
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "type": "milvus",
            "title": "H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf",
            "score": 0.95,
            "text": """
             LA2608安装SIM卡后，通过3G/4G网络连接运营商无线控制器。
        需要配置Cellular接口、自动拨号、DNS解析、默认路由，
        并通过wlan ac ip命令指定无线控制器地址。
           具体组网方式请参考下图：
          ![LA2608组网示意图](http://127.0.0.1:9000/knowledge-base-files/upload-images/H3C%20LA2608室内无线网关%20用户手册-6W100-整本手册/afb51b65a297a9e57d800efeecc46d72840ae5f0b9c2dc62f843b163efb45ec0.jpg)
nikan            """,
        }
    ]
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？", "rewritten_query": "H3C LA2608室内无线网关的具体操作步骤和面板设置方法"},
    ]
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "H3C LA2608室内无线网关怎么操作？",
        "rewritten_query": "H3C LA2608室内无线网关的具体操作步骤和面板设置方法",
        "item_names": ["H3C LA2608室内无线网关"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,
        "answer": None,
    }
    result = node_answer_output(mock_state)
    print(result)