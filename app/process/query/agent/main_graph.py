from langgraph.graph import END, StateGraph

from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_agent_execute import node_agent_execute
from app.process.query.agent.nodes.node_clarify import node_clarify
from app.process.query.agent.nodes.node_direct_answer import node_direct_answer
from app.process.query.agent.nodes.node_evidence_grader import node_evidence_grader
from app.process.query.agent.nodes.node_evidence_response import node_evidence_response
from app.process.query.agent.nodes.node_item_name_confim import node_item_name_confirm
from app.process.query.agent.nodes.node_load_context import node_load_context
from app.process.query.agent.nodes.node_query_planner import node_query_planner
from app.process.query.agent.nodes.node_resolve_reference import node_resolve_reference
from app.process.query.agent.nodes.node_prepare_evidence_retry import node_prepare_evidence_retry
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.routing import (
    route_after_item_confirmation,
    route_after_reference_resolution,
    route_after_evidence_grader,
    route_after_planner,
    route_after_web_search,
    route_evidence_retry_tools,
)
from app.process.query.agent.state import QueryGraphState


query_graph_builder = StateGraph(state_schema=QueryGraphState)

query_graph_builder.add_node(node_load_context)
query_graph_builder.add_node(node_query_planner)
query_graph_builder.add_node(node_resolve_reference)
query_graph_builder.add_node(node_direct_answer)
query_graph_builder.add_node(node_clarify)
query_graph_builder.add_node(node_evidence_grader)
query_graph_builder.add_node(node_evidence_response)
query_graph_builder.add_node(node_prepare_evidence_retry)
query_graph_builder.add_node(node_item_name_confirm)
query_graph_builder.add_node(node_search_embedding)
query_graph_builder.add_node(node_search_embedding_hyde)
query_graph_builder.add_node(node_web_search_mcp)
query_graph_builder.add_node(node_rrf)
query_graph_builder.add_node(node_rerank)
query_graph_builder.add_node(node_answer_output)
query_graph_builder.add_node(node_agent_execute)

# 所有请求先读取历史、解析指代，再由 Planner 生成结构化决策。
query_graph_builder.set_entry_point("node_load_context")
query_graph_builder.add_edge("node_load_context", "node_resolve_reference")
query_graph_builder.add_conditional_edges(
    "node_resolve_reference",
    route_after_reference_resolution,
    {
        "node_query_planner": "node_query_planner",
        "node_clarify": "node_clarify",
    },
)

query_graph_builder.add_conditional_edges(
    "node_query_planner",
    route_after_planner,
    {
        "node_direct_answer": "node_direct_answer",
        "node_clarify": "node_clarify",
        "node_item_name_confirm": "node_item_name_confirm",
        "node_web_search_mcp": "node_web_search_mcp",
        "node_agent_execute": "node_agent_execute",
    },
)

query_graph_builder.add_edge("node_direct_answer", "node_answer_output")
query_graph_builder.add_edge("node_clarify", "node_answer_output")

query_graph_builder.add_conditional_edges(
    "node_item_name_confirm",
    route_after_item_confirmation,
    {
        "node_answer_output": "node_answer_output",
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp",
        "node_agent_execute": "node_agent_execute",
    },
)

query_graph_builder.add_edge("node_agent_execute", "node_answer_output")

# 本地两路结果在 RRF 汇合；纯 Web 跳过 RRF，混合 Web 结果进入 RRF 汇合点。
query_graph_builder.add_edge("node_search_embedding", "node_rrf")
query_graph_builder.add_edge("node_search_embedding_hyde", "node_rrf")
query_graph_builder.add_conditional_edges(
    "node_web_search_mcp",
    route_after_web_search,
    {
        "node_rrf": "node_rrf",
        "node_rerank": "node_rerank",
    },
)

query_graph_builder.add_edge("node_rrf", "node_rerank")
query_graph_builder.add_edge("node_rerank", "node_evidence_grader")
query_graph_builder.add_conditional_edges(
    "node_evidence_grader",
    route_after_evidence_grader,
    {
        "node_answer_output": "node_answer_output",
        "node_prepare_evidence_retry": "node_prepare_evidence_retry",
        "node_evidence_response": "node_evidence_response",
    },
)
query_graph_builder.add_conditional_edges(
    "node_prepare_evidence_retry",
    route_evidence_retry_tools,
    {
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp",
        "node_evidence_response": "node_evidence_response",
    },
)
query_graph_builder.add_edge("node_evidence_response", "node_answer_output")
query_graph_builder.add_edge("node_answer_output", END)

query_graph_app = query_graph_builder.compile()
# ASCII 可视化
print("\n===== 图结构 ASCII =====")
query_graph_app.get_graph().print_ascii()

# Mermaid 源码
print("\n===== Mermaid 代码 =====")
print(query_graph_app.get_graph().draw_mermaid())
