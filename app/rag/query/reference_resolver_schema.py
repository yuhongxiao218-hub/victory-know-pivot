from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ReferenceStatus = Literal["resolved", "ambiguous", "not_required", "missing"]
ReferenceSource = Literal["explicit", "history", "none"]


class ReferenceResolution(BaseModel):
    """会话指代解析结果，作为 Resolver 与 Router 之间的数据契约。"""

    model_config = ConfigDict(extra="forbid")

    status: ReferenceStatus
    item_names: list[str] = Field(default_factory=list)
    standalone_query: str
    source: ReferenceSource = "none"
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)
