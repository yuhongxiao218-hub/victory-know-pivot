from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.query.planner_schema import PlannerTool


EvidenceAction = Literal["answer", "retry", "clarify", "refuse", "fail"]


class EvidenceGrade(BaseModel):
    """Evidence Grader 与 LangGraph Router 之间的数据契约。"""

    model_config = ConfigDict(extra="forbid")

    action: EvidenceAction
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)
    retry_tools: list[PlannerTool] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_retry_tools(self) -> "EvidenceGrade":
        if self.action == "retry" and not self.retry_tools:
            raise ValueError("retry 动作必须提供 retry_tools")
        if self.action != "retry" and self.retry_tools:
            raise ValueError(f"{self.action} 动作不应提供 retry_tools")
        if len(set(self.retry_tools)) != len(self.retry_tools):
            raise ValueError("retry_tools 不应包含重复工具")
        return self
