"""base.py – Abstract base class for all bandit policies.

Defines the interface that every bandit algorithm must implement:
select_arm(context), update(context, arm, reward), and reset().
"""
