from app.process.query.agent.routing import (
    route_after_evidence_grader,
    route_after_item_confirmation,
    route_after_reference_resolution,
    route_after_planner,
    route_after_web_search,
    route_evidence_retry_tools,
)


def test_reference_resolution_routes_before_planner():
    assert route_after_reference_resolution({"reference_status": "resolved"}) == "node_query_planner"
    assert route_after_reference_resolution({"reference_status": "not_required"}) == "node_query_planner"
    assert route_after_reference_resolution({"reference_status": "ambiguous"}) == "node_clarify"
    assert route_after_reference_resolution({"reference_status": "missing"}) == "node_clarify"


def test_planner_routes_map_to_expected_nodes():
    assert route_after_planner({"planner_route": "direct_answer"}) == "node_direct_answer"
    assert route_after_planner({"planner_route": "clarify"}) == "node_clarify"
    assert route_after_planner({"planner_route": "device_rag"}) == "node_item_name_confirm"
    assert route_after_planner({"planner_route": "web_rag"}) == "node_web_search_mcp"
    assert route_after_planner({"planner_route": "hybrid_rag"}) == "node_item_name_confirm"


def test_multi_step_plan_routes_through_agent():
    web_plan = {"steps": [{"tool": "web_search"}, {"tool": "compare_products"}]}
    local_plan = {"steps": [{"tool": "local_search"}, {"tool": "compare_products"}]}

    assert route_after_planner({"task_plan": web_plan}) == "node_agent_execute"
    assert route_after_planner({"task_plan": local_plan}) == "node_item_name_confirm"
    assert route_after_item_confirmation({"task_plan": local_plan, "item_names": ["设备A"]}) == "node_agent_execute"


def test_missing_item_name_ends_with_existing_clarification_answer():
    state = {"item_names": [], "selected_tools": ["local_search"]}
    assert route_after_item_confirmation(state) == "node_answer_output"


def test_item_confirmation_starts_only_selected_tools():
    state = {
        "item_names": ["H3C LA2608"],
        "selected_tools": ["local_search", "web_search"],
    }
    assert route_after_item_confirmation(state) == (
        "node_search_embedding",
        "node_web_search_mcp",
    )


def test_web_only_skips_rrf_but_hybrid_web_joins_rrf():
    assert route_after_web_search({"planner_route": "web_rag"}) == "node_rerank"
    assert route_after_web_search({"planner_route": "hybrid_rag"}) == "node_rrf"


def test_evidence_action_routes_to_expected_node():
    assert route_after_evidence_grader({"evidence_action": "answer"}) == "node_answer_output"
    assert route_after_evidence_grader({"evidence_action": "retry"}) == "node_prepare_evidence_retry"
    assert route_after_evidence_grader({"evidence_action": "clarify"}) == "node_evidence_response"
    assert route_after_evidence_grader({"evidence_action": "refuse"}) == "node_evidence_response"
    assert route_after_evidence_grader({"evidence_action": "fail"}) == "node_evidence_response"


def test_evidence_retry_routes_selected_tools():
    assert route_evidence_retry_tools(
        {"selected_tools": ["hyde_search", "web_search"]}
    ) == ("node_search_embedding_hyde", "node_web_search_mcp")
