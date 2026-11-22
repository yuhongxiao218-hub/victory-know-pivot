from typing_extensions import TypedDict
from typing import Annotated, List
import copy


def merge_tool_executions(left: list[dict], right: list[dict]) -> list[dict]:
    """按 execution_id 合并工具记录，避免循环中完整 State 被重复累计。"""
    merged: list[dict] = []
    seen_execution_ids: set[str] = set()
    for record in [*(left or []), *(right or [])]:
        execution_id = record.get("execution_id")
        if execution_id and execution_id in seen_execution_ids:
            continue
        merged.append(record)
        if execution_id:
            seen_execution_ids.add(execution_id)
    return merged

class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    使用字典式访问（如 state["session_id"]、state.get("answer")）。
    """
    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记
    image_urls: List[str]  # 答案中引用的图片链接

    # 会话指代解析结果
    reference_status: str
    reference_source: str
    reference_confidence: float
    reference_reason: str
    reference_candidates: List[str]

    # Agent Planner 的结构化决策结果
    planner_intent: str  # 用户意图，如 chat / device_knowledge
    planner_route: str  # Planner 选择的主处理路径
    actual_route: str  # Graph 实际执行的主路径；未经过 Planner 的分支也必须记录
    selected_tools: List[str]  # Planner 计划调用的工具名称
    planner_confidence: float  # 模型自评置信度，仅用于观测
    planner_reason: str  # 简短、可记录的决策理由

    # 多个并行工具都向这个字段追加记录；自定义 Reducer 会按执行ID去重。
    tool_executions: Annotated[List[dict], merge_tool_executions]

    # 多步 Agent 运行状态
    task_plan: dict
    current_step_index: int
    step_observations: List[dict]
    step_evaluation: dict
    agent_step_count: int
    agent_max_steps: int
    agent_replan_count: int
    agent_max_replans: int
    agent_tool_failure_count: int
    agent_max_tool_failures: int
    agent_max_repeated_calls: int
    agent_status: str
    agent_finish_reason: str

    # Evidence Grader 的评估与有限重试状态
    evidence_action: str
    evidence_confidence: float
    evidence_reason: str
    evidence_retry_tools: List[str]
    evidence_retry_count: int


# ========================
# 默认状态（全部为空）
# ========================
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "image_urls": [],
    "reference_status": "",
    "reference_source": "none",
    "reference_confidence": 0.0,
    "reference_reason": "",
    "reference_candidates": [],
    "planner_intent": "",
    "planner_route": "",
    "actual_route": "",
    "selected_tools": [],
    "planner_confidence": 0.0,
    "planner_reason": "",
    "tool_executions": [],
    "task_plan": {},
    "current_step_index": 0,
    "step_observations": [],
    "step_evaluation": {},
    "agent_step_count": 0,
    "agent_max_steps": 6,
    "agent_replan_count": 0,
    "agent_max_replans": 2,
    "agent_tool_failure_count": 0,
    "agent_max_tool_failures": 2,
    "agent_max_repeated_calls": 1,
    "agent_status": "idle",
    "agent_finish_reason": "",
    "evidence_action": "",
    "evidence_confidence": 0.0,
    "evidence_reason": "",
    "evidence_retry_tools": [],
    "evidence_retry_count": 0,
}


# ========================
# 创建默认状态（可覆盖）
# ========================
def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


# ========================
# 获取干净状态
# ========================
def get_query_default_state() -> QueryGraphState:
    """
    返回一个新的状态实例，避免全局变量污染。
    """
    return copy.deepcopy(query_graph_default_state)


# ========================
# 状态复制函数
# ========================
def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    复制现有状态并可覆盖字段，深拷贝，不污染原数据
    """
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == "__main__":
    # 测试
    state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样?",
        is_stream=False
    )
    print("初始化状态：", state)

    # 复制状态
    new_state = copy_query_state(
        state,
        original_query="修改后的问题"
    )
    print("复制后的状态：", new_state)
