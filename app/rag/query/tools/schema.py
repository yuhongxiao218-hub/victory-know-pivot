from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.query.planner_schema import PlannerTool


class SearchToolResult(BaseModel):
    """所有检索工具统一返回的数据结构。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: PlannerTool
    execution_id: str = Field(default_factory=lambda: uuid4().hex)
    success: bool
    documents: list[dict[str, Any]] = Field(default_factory=list)
    duration_ms: int = Field(ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_error_state(self) -> "SearchToolResult":
        if self.success and self.error:
            raise ValueError("工具执行成功时不应包含 error")
        if not self.success and not self.error:
            raise ValueError("工具执行失败时必须包含 error")
        return self

    def to_trace(self) -> dict[str, Any]:
        """生成写入 Graph State 的轻量执行记录，避免重复存储文档正文。"""
        return {
            "tool_name": self.tool_name,
            "execution_id": self.execution_id,
            "success": self.success,
            "result_count": len(self.documents),
            "duration_ms": self.duration_ms,
            "error": self.error,
        }
