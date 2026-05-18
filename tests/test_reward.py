import unittest

from src.bandits.epsilon_greedy import EpsilonGreedyBandit
from src.bandits.thompson import ThompsonSampling
from src.bandits.ucb import UCBBandit
from src.reward import RewardBuffer, RewardEngine


class TestRewardEngine(unittest.TestCase):
    def test_fee_calculation_is_1_point_8_percent(self) -> None:
        self.assertEqual(RewardEngine.calculate_immediate_fee(10000.0), 180.0)

    def test_default_penalty_is_full_outstanding_amount_negative(self) -> None:
        self.assertEqual(RewardEngine.calculate_default_penalty(7250.0, True), -7250.0)

    def test_non_default_has_zero_penalty(self) -> None:
        self.assertEqual(RewardEngine.calculate_default_penalty(7250.0, False), 0.0)


class TestRewardBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = RewardBuffer(delay_months=3)
        self.context = [0.1, 0.2, 0.3]
        self.buffer.record_action("USER_00001", 1, "plus_10", self.context)
        self.buffer.receive_outcome(
            user_id="USER_00001",
            month=1,
            amount_spent=10000.0,
            outstanding_amount=5000.0,
            did_default=False,
        )

    def test_rewards_are_not_returned_before_delay(self) -> None:
        ready_rewards = self.buffer.get_ready_rewards(current_month=3)
        self.assertEqual(ready_rewards, [])

    def test_rewards_are_returned_exactly_at_t_plus_3(self) -> None:
        ready_rewards = self.buffer.get_ready_rewards(current_month=4)
        self.assertEqual(len(ready_rewards), 1)
        self.assertEqual(ready_rewards[0]["user_id"], "USER_00001")
        self.assertEqual(ready_rewards[0]["action_month"], 1)
        self.assertEqual(ready_rewards[0]["action"], "plus_10")
        self.assertEqual(ready_rewards[0]["reward"], 180.0)
        self.assertFalse(ready_rewards[0]["is_default"])

    def test_pending_count_decreases_when_rewards_become_ready(self) -> None:
        self.assertEqual(self.buffer.pending_count(), 1)
        self.buffer.get_ready_rewards(current_month=3)
        self.assertEqual(self.buffer.pending_count(), 1)
        self.buffer.get_ready_rewards(current_month=4)
        self.assertEqual(self.buffer.pending_count(), 0)


class TestBanditNormalization(unittest.TestCase):
    def test_thompson_two_branch_normalization(self) -> None:
        ts = ThompsonSampling()
        self.assertEqual(ts.normalize_reward(700_000), 1.0)
        self.assertGreater(ts.normalize_reward(350_000), 0.9)
        self.assertEqual(ts.normalize_reward(0), 0.5)
        self.assertLess(ts.normalize_reward(-13_500_000), 0.1)
        self.assertEqual(ts.normalize_reward(-27_000_000), 0.0)
        self.assertGreater(ts.normalize_reward(-26_045_436), 0.0)
        self.assertLess(ts.normalize_reward(-26_045_436), 0.02)
        self.assertGreater(ts.normalize_reward(3_655), 0.8)
        self.assertLess(ts.normalize_reward(3_655), 0.81)

    def test_ucb_symmetric_normalization(self) -> None:
        ucb = UCBBandit()
        self.assertEqual(ucb._normalize(27_000_000), 1.0)
        self.assertEqual(ucb._normalize(-27_000_000), -1.0)
        self.assertEqual(ucb._normalize(0), 0.0)
        self.assertGreater(ucb._normalize(3_655), 0.47)
        self.assertLess(ucb._normalize(3_655), 0.49)

    def test_epsilon_greedy_symmetric_normalization(self) -> None:
        bandit = EpsilonGreedyBandit()
        self.assertEqual(bandit._normalize(27_000_000), 1.0)
        self.assertEqual(bandit._normalize(-27_000_000), -1.0)
        self.assertEqual(bandit._normalize(0), 0.0)


if __name__ == "__main__":
    unittest.main()
