"""查询 Agent 使用的标准化工具适配层。"""

from app.rag.query.tools.registry import execute_search_tool, get_search_tool, list_search_tools
from app.rag.query.tools.schema import SearchToolResult

__all__ = [
    "SearchToolResult",
    "execute_search_tool",
    "get_search_tool",
    "list_search_tools",
]
