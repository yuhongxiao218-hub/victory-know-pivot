import unittest

from app.process.query.agent.state import create_query_default_state


class AgentStateTest(unittest.TestCase):
    def test_agent_runtime_defaults_are_ready_for_a_new_plan(self):
        state = create_query_default_state()

        self.assertEqual(state["task_plan"], {})
        self.assertEqual(state["current_step_index"], 0)
        self.assertEqual(state["step_observations"], [])
        self.assertEqual(state["step_evaluation"], {})
        self.assertEqual(state["agent_step_count"], 0)
        self.assertEqual(state["agent_max_steps"], 6)
        self.assertEqual(state["agent_replan_count"], 0)
        self.assertEqual(state["agent_max_replans"], 2)
        self.assertEqual(state["agent_tool_failure_count"], 0)
        self.assertEqual(state["agent_max_tool_failures"], 2)
        self.assertEqual(state["agent_max_repeated_calls"], 1)
        self.assertEqual(state["agent_status"], "idle")
        self.assertEqual(state["agent_finish_reason"], "")


if __name__ == "__main__":
    unittest.main()
