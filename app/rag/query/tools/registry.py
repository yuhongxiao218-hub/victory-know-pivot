import time
from collections.abc import Callable
from dataclasses import dataclass

from app.process.query.agent.state import QueryGraphState
from app.rag.query.embedding_search_service import search_by_embedding
from app.rag.query.hyde_search_service import search_embedding_hyde
from app.rag.query.planner_schema import PlannerTool
from app.rag.query.product_compare_service import compare_products
from app.rag.query.tools.schema import SearchToolResult
from app.rag.query.web_search_service import search_by_web
from app.shared.runtime.logger import logger, step_log


SearchToolHandler = Callable[[QueryGraphState], list[dict]]


@dataclass(frozen=True)
class SearchToolDefinition:
    """工具注册信息：名称用于路由，描述用于规划，handler 调用真实 Service。"""

    name: PlannerTool
    description: str
    requires_item_name: bool
    handler: SearchToolHandler


SEARCH_TOOL_REGISTRY: dict[PlannerTool, SearchToolDefinition] = {
    "local_search": SearchToolDefinition(
        name="local_search",
        description="在本地设备知识库中执行 Dense/Sparse 混合检索",
        requires_item_name=True,
        handler=search_by_embedding,
    ),
    "hyde_search": SearchToolDefinition(
        name="hyde_search",
        description="先生成假设答案，再到本地设备知识库执行语义检索",
        requires_item_name=True,
        handler=search_embedding_hyde,
    ),
    "web_search": SearchToolDefinition(
        name="web_search",
        description="搜索互联网中的最新设备与时效信息",
        requires_item_name=False,
        handler=search_by_web,
    ),
    # shortcut: 沿用 SearchTool 旧接口承载行动工具，继续扩展工具类型时统一重命名。
    "compare_products": SearchToolDefinition(
        name="compare_products",
        description="从已收集证据中抽取产品参数、差异和选择建议",
        requires_item_name=False,
        handler=compare_products,
    ),
}


def get_search_tool(tool_name: PlannerTool) -> SearchToolDefinition:
    """根据 Planner 输出的稳定工具名取得工具定义。"""
    try:
        return SEARCH_TOOL_REGISTRY[tool_name]
    except KeyError as exc:
        raise ValueError(f"未注册的Agent工具: {tool_name}") from exc


def list_search_tools() -> list[dict]:
    """提供可供 Planner、接口文档或调试页面读取的工具清单。"""
    return [
        {
            "name": definition.name,
            "description": definition.description,
            "requires_item_name": definition.requires_item_name,
        }
        for definition in SEARCH_TOOL_REGISTRY.values()
    ]


@step_log("execute_search_tool")
def execute_search_tool(tool_name: PlannerTool, state: QueryGraphState) -> SearchToolResult:
    """统一执行 Agent 工具，将异常转换为可观测的结构化结果。"""
    definition = get_search_tool(tool_name)
    start_time = time.perf_counter()

    if definition.requires_item_name and not state.get("item_names"):
        return SearchToolResult(
            tool_name=tool_name,
            success=False,
            documents=[],
            duration_ms=0,
            error=f"{tool_name} 需要已经确认的 item_names",
        )

    try:
        documents = definition.handler(state)
        if not isinstance(documents, list):
            raise TypeError(f"{tool_name} 必须返回 list，实际为 {type(documents).__name__}")
        return SearchToolResult(
            tool_name=tool_name,
            success=True,
            documents=documents,
            duration_ms=int((time.perf_counter() - start_time) * 1000),
        )
    except Exception as exc:
        logger.exception(f"Agent工具执行失败: tool={tool_name}")
        return SearchToolResult(
            tool_name=tool_name,
            success=False,
            documents=[],
            duration_ms=int((time.perf_counter() - start_time) * 1000),
            error=str(exc),
        )
