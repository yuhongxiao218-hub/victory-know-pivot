from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PlannerIntent = Literal[
    "chat",
    "device_knowledge",
    "latest_information",
    "ambiguous",
    "unsupported",
]

PlannerRoute = Literal[
    "direct_answer",
    "device_rag",
    "web_rag",
    "hybrid_rag",
    "clarify",
]

PlannerTool = Literal[
    "local_search",
    "hyde_search",
    "web_search",
    "compare_products",
]


class TaskPlanStep(BaseModel):
    """复杂目标中的一个顺序执行步骤。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=50)
    description: str = Field(min_length=1, max_length=300)
    tool: PlannerTool
    success_criteria: str = Field(min_length=1, max_length=300)


class TaskPlan(BaseModel):
    """Planner 为复杂目标生成的最小多步计划契约。"""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=500)
    steps: list[TaskPlanStep] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_unique_step_ids(self) -> "TaskPlan":
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("任务计划中的步骤ID不应重复")
        return self


class PlannerDecision(BaseModel):
    """Planner 与 LangGraph Router 之间的结构化数据契约。"""

    model_config = ConfigDict(extra="forbid")

    intent: PlannerIntent
    route: PlannerRoute
    selected_tools: list[PlannerTool] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)
    task_plan: TaskPlan | None = None

    @model_validator(mode="after")
    def validate_route_tools(self) -> "PlannerDecision":
        """阻止主路由与工具列表互相矛盾。"""
        tools = set(self.selected_tools)

        if len(tools) != len(self.selected_tools):
            raise ValueError("selected_tools 不应包含重复工具")

        if self.route in {"direct_answer", "clarify"} and tools:
            raise ValueError(f"{self.route} 路由不应调用检索工具")
        if self.route == "device_rag":
            if "local_search" not in tools:
                raise ValueError("device_rag 路由必须包含 local_search")
            if not tools.issubset({"local_search", "hyde_search"}):
                raise ValueError("device_rag 路由只能使用本地检索和HyDE工具")
        if self.route == "web_rag" and tools != {"web_search"}:
            raise ValueError("web_rag 路由只能使用 web_search")
        if self.route == "hybrid_rag":
            if not {"local_search", "web_search"}.issubset(tools):
                raise ValueError("hybrid_rag 路由必须同时包含 local_search 和 web_search")
            if not tools.issubset({"local_search", "hyde_search", "web_search"}):
                raise ValueError("hybrid_rag 路由只能使用检索工具")

        allowed_routes_by_intent = {
            "chat": {"direct_answer"},
            "device_knowledge": {"device_rag", "hybrid_rag"},
            "latest_information": {"web_rag", "hybrid_rag"},
            "ambiguous": {"clarify"},
            "unsupported": {"direct_answer"},
        }
        if self.route not in allowed_routes_by_intent[self.intent]:
            raise ValueError(f"intent={self.intent} 与 route={self.route} 不一致")

        if self.task_plan is not None:
            if self.route in {"direct_answer", "clarify"}:
                raise ValueError(f"{self.route} 路由不应生成多步任务计划")
            if len(self.task_plan.steps) < 2:
                raise ValueError("初始多步任务计划至少需要两个步骤")
            planned_search_tools = {
                step.tool for step in self.task_plan.steps if step.tool != "compare_products"
            }
            if planned_search_tools != tools:
                raise ValueError("任务计划中的检索工具必须与 selected_tools 一致")

        return self


def default_planner_decision(reason: str) -> PlannerDecision:
    """Planner 不可用时退回当前项目已有的本地 RAG 主路径。"""
    return PlannerDecision(
        intent="device_knowledge",
        route="device_rag",
        selected_tools=["local_search"],
        confidence=0.0,
        reason=reason,
    )
