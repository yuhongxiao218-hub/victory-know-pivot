import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.process.query.agent.state import QueryGraphState
from app.rag.query.evidence_grader_service import format_evidence, invoke_grader_model
from app.rag.query.planner_schema import TaskPlan
from app.shared.runtime.load_prompt import load_prompt


class ProductFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    parameters: dict[str, str]


class ProductComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    products: list[ProductFacts] = Field(min_length=2, max_length=6)
    differences: list[str] = Field(min_length=1)
    recommendation: str = Field(min_length=1, max_length=1000)
    missing_information: list[str] = Field(default_factory=list)


ComparisonInvoker = Callable[[str], dict[str, Any]]


def compare_products(
    state: QueryGraphState,
    comparison_invoker: ComparisonInvoker | None = None,
) -> list[dict]:
    """从前序步骤证据中抽取产品参数并生成结构化对比。"""
    plan = TaskPlan.model_validate(state.get("task_plan"))
    documents = [
        document
        for observation in state.get("step_observations", [])
        if observation.get("success")
        for document in observation.get("documents", [])
    ]
    if not documents:
        raise ValueError("产品对比需要前序步骤提供证据")

    prompt = load_prompt(
        "product_compare",
        goal=plan.goal,
        evidence=format_evidence(documents, limit=10),
    )
    comparison = ProductComparison.model_validate(
        (comparison_invoker or invoke_grader_model)(prompt)
    )
    comparison_data = comparison.model_dump()
    return [
        {
            "title": "产品参数对比",
            "type": "action",
            "content": json.dumps(comparison_data, ensure_ascii=False),
            "comparison": comparison_data,
        }
    ]
