import re

from app.process.query.agent.state import QueryGraphState
from app.rag.query.reference_resolver_schema import ReferenceResolution


_REFERENCE_PATTERN = re.compile(
    r"(?:这|那)(?:台|款)(?:机器|设备|产品)?"
    r"|(?:这|那)个(?:机器|设备|产品)?"
    r"|该(?:机器|设备|产品)"
    r"|它们?"
    r"|本机"
)

_MODEL_PATTERN = re.compile(
    r"(?i)(?<![A-Z0-9])(?=[A-Z0-9._-]*[A-Z])(?=[A-Z0-9._-]*\d)[A-Z0-9][A-Z0-9._-]{2,}(?![A-Z0-9])"
)


def recent_active_item_names(history: list[dict]) -> list[str]:
    """返回最近一条已确认用户消息的主体；上层历史按时间正序排列。"""
    for message in reversed(history):
        if message.get("role") != "user":
            continue
        item_names = [str(name).strip() for name in message.get("item_names") or [] if str(name).strip()]
        if item_names:
            return list(dict.fromkeys(item_names))
    return []


def has_explicit_subject_hint(query: str, active_item_names: list[str]) -> bool:
    """模型号或完整历史主体已出现在问题中时，不把它误判成纯指代问题。"""
    normalized_query = query.casefold()
    if any(name.casefold() in normalized_query for name in active_item_names):
        return True
    return bool(_MODEL_PATTERN.search(query))


def rewrite_reference_query(query: str, item_name: str) -> str:
    """只替换第一个指代表达，保留用户原问题的其他约束。"""
    return _REFERENCE_PATTERN.sub(item_name, query, count=1)


def resolve_reference(state: QueryGraphState) -> ReferenceResolution:
    """使用已确认会话主体解析代词；不重新调用 LLM 或向量库。"""
    query = state.get("original_query", "").strip()
    active_item_names = recent_active_item_names(state.get("history", []))
    contains_reference = bool(_REFERENCE_PATTERN.search(query))
    explicit_subject = has_explicit_subject_hint(query, active_item_names)

    if not contains_reference or explicit_subject:
        return ReferenceResolution(
            status="not_required",
            item_names=[],
            standalone_query=query,
            source="explicit" if explicit_subject else "none",
            confidence=1.0,
            reason="问题不需要历史指代解析",
        )

    if not active_item_names:
        return ReferenceResolution(
            status="missing",
            item_names=[],
            standalone_query=query,
            source="none",
            confidence=1.0,
            reason="问题包含指代表达，但当前会话没有已确认设备",
        )

    if len(active_item_names) > 1:
        return ReferenceResolution(
            status="ambiguous",
            item_names=active_item_names,
            standalone_query=query,
            source="history",
            confidence=1.0,
            reason="最近一轮关联多个设备，无法唯一确定指代对象",
        )

    item_name = active_item_names[0]
    return ReferenceResolution(
        status="resolved",
        item_names=[item_name],
        standalone_query=rewrite_reference_query(query, item_name),
        source="history",
        confidence=0.98,
        reason=f"沿用最近一轮已确认主体：{item_name}",
    )
