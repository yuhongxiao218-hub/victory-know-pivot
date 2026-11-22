import unittest

from pydantic import ValidationError

from app.rag.query.planner_schema import TaskPlan


class TaskPlanSchemaTest(unittest.TestCase):
    def test_accepts_ordered_multi_step_plan(self):
        plan = TaskPlan.model_validate(
            {
                "goal": "比较两款设备并给出选择建议",
                "steps": [
                    {
                        "id": "step_1",
                        "description": "检索第一款设备的参数",
                        "tool": "local_search",
                        "success_criteria": "获得可引用的设备参数",
                    },
                    {
                        "id": "step_2",
                        "description": "补充第二款设备的最新信息",
                        "tool": "web_search",
                        "success_criteria": "获得有来源的最新信息",
                    },
                ],
            }
        )

        self.assertEqual(plan.goal, "比较两款设备并给出选择建议")
        self.assertEqual([step.id for step in plan.steps], ["step_1", "step_2"])

    def test_rejects_empty_or_duplicate_steps(self):
        with self.assertRaises(ValidationError):
            TaskPlan(goal="比较设备", steps=[])

        duplicate_step = {
            "id": "step_1",
            "description": "检索设备参数",
            "tool": "local_search",
            "success_criteria": "获得设备参数",
        }
        with self.assertRaises(ValidationError):
            TaskPlan(goal="比较设备", steps=[duplicate_step, duplicate_step])


if __name__ == "__main__":
    unittest.main()
