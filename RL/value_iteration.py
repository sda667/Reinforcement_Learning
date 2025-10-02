import numpy as np
from .env import Labyrinth
from collections import defaultdict


class ValueIteration:
    """
    Value Iteration algorithm for solving a reinforcement learning environment.
    The algorithm iteratively updates the estimated values of states to find an optimal policy.

    Attributes:
    - env (Labyrinth): The environment in which the agent operates.
    - gamma (float): Discount factor for future rewards.
    ...
    """

    def __init__(self, env: Labyrinth, gamma: float = .95):
        """
        Initialize the Value Iteration agent with specified parameters.

        Parameters:
        - env (Labyrinth): The environment in which the agent operates.
        - gamma (float): Discount factor (0 < gamma <= 1) for future rewards.
        """
        self.env = env
        self.gamma = gamma
        height, width = env.get_map_size()
        self.values = np.zeros((height, width))

    def train(self, delta: float, max_iterations: int = 1000):
        """
        Train the agent using value iteration for a specified number of updates.

        Parameters:
        - delta: Minimum update size required to continue updating
        """
        for _ in range(max_iterations):
            delta_max = 0
            new_values = np.copy(self.values)

            for state in self.env.get_valid_states():
                q_values = []

                for action in self.env.get_all_actions():
                    # --- save current env state ---
                    old_state = self.env.get_observation()
                    old_done = self.env.is_done()

                    # --- simulate step ---
                    reward = self.env.step(action, state=state)
                    next_state = self.env.get_observation()

                    # --- update Q ---
                    q = reward + self.gamma * self.values[next_state]
                    q_values.append(q)

                    # --- restore env ---
                    self.env.set_state(old_state)
                    self.env._done = old_done

                if q_values:
                    new_values[state] = max(q_values)

                delta_max = max(delta_max, abs(new_values[state] - self.values[state]))

            self.values = new_values
            if delta_max < delta:
                break

    def get_value_table(self) -> np.ndarray:
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
        - np.ndarray: A 2D array representing the estimated values for each state.
        """
        return self.values
