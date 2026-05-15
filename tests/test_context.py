import unittest

import numpy as np

from src.context import ContextBuilder
from src.simulator import generate_users


class TestContextBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.users_df = generate_users(n=25, seed=7)
        cls.builder = ContextBuilder(cls.users_df)
        cls.user_id = cls.users_df.iloc[0]["user_id"]
        cls.sample_history = [
            {
                "month": 1,
                "amount_spent": 18000.0,
                "outstanding_amount": 4000.0,
                "current_credit_limit": 50000.0,
            },
            {
                "month": 2,
                "amount_spent": 22000.0,
                "outstanding_amount": 6000.0,
                "current_credit_limit": 50000.0,
            },
            {
                "month": 3,
                "amount_spent": 20000.0,
                "outstanding_amount": 5500.0,
                "current_credit_limit": 50000.0,
            },
        ]

    def test_context_vector_shape_is_ten(self) -> None:
        context = self.builder.build_context(self.user_id, self.sample_history)
        self.assertEqual(context.shape, (10,))

    def test_all_context_values_are_bounded(self) -> None:
        context = self.builder.build_context(self.user_id, self.sample_history)
        self.assertTrue(np.all(context >= 0.0))
        self.assertTrue(np.all(context <= 1.0))
        self.assertFalse(np.isnan(context).any())
        self.assertFalse(np.isinf(context).any())

    def test_apply_action_plus_10_is_exact(self) -> None:
        self.assertEqual(ContextBuilder.apply_action(100000.0, "plus_10"), 110000.0)

    def test_cold_start_returns_valid_vector(self) -> None:
        context = self.builder.build_context(self.user_id, [])
        self.assertEqual(context.shape, (10,))
        self.assertTrue(np.all(context >= 0.0))
        self.assertTrue(np.all(context <= 1.0))

    def test_same_inputs_are_deterministic(self) -> None:
        first = self.builder.build_context(self.user_id, self.sample_history)
        second = self.builder.build_context(self.user_id, self.sample_history)
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
