from .env import Labyrinth

import numpy as np
import random
from collections import defaultdict

class QLearning:
    """
    Q-Learning algorithm for training an agent in a given environment.
    The agent learns an optimal policy for selecting actions to maximize cumulative rewards.

    Attributes:
    - env (Labyrinth): The environment in which the agent operates.
    - gamma (float): Discount factor for future rewards.
    - alpha (float): Learning rate.
    - epsilon (float): Probability of taking a random action (exploration).
    - c (float): Parameter for exploration/exploitation balance in action selection.
    ...
    """

    def __init__(self, env: Labyrinth, gamma: float = 0.9, alpha: float = 0.1, epsilon: float = 0, tau: float = 0):
        """
        Initialize the Q-Learning agent with specified parameters.

        Parameters:
        - env (Labyrinth): The environment in which the agent operates.
        - gamma (float): Discount factor (0 < gamma <= 1) for future rewards.
        - alpha (float): Learning rate (0 < alpha <= 1) for updating Q-values.
        - epsilon (float): Probability (0 <= epsilon <= 1) for exploration in action selection.
        - tau (float): Exploration temperature.
        """
        self.env = env
        self.gamma = gamma          
        self.alpha = alpha          
        self.epsilon = epsilon      
        self.tau = tau
        height, width = env.get_map_size()
        no_actions = len(env.get_all_actions())
        self.q_table = np.zeros((height, width, no_actions))  # Initialize Q-table with zeros

    def get_q_table(self) -> np.ndarray:
        """
        Retrieve the Q-table as a 3D numpy array for visualization.

        Returns:
        - np.ndarray: A 3D array representing Q-values for each state-action pair.
        """
        return self.q_table

    def choose_action(self, state):
        if self.epsilon > 0:  # ε-greedy
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.env.get_all_actions())
            return np.argmax(self.q_table[state])

        if self.tau > 0:  # Softmax
            preferences = self.q_table[state] / self.tau
            exp_prefs = np.exp(preferences - np.max(preferences))
            probs = exp_prefs / np.sum(exp_prefs)
            return np.random.choice(self.env.get_all_actions(), p=probs)

        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (0 if done else self.gamma * self.q_table[next_state][best_next_action])
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def train(self, n_steps: int):
        """
        Train the Q-learning agent over a specified number of steps.

        Parameters:
        - n_steps (int): Total number of steps for training.
        """
        state = self.env.reset()
        for step in range(n_steps):
            action = self.choose_action(state)

            reward = self.env.step(action)
            next_state = self.env.get_observation()
            done = self.env.is_done()

            self.update(state, action, reward, next_state, done)
            state = next_state

            if done:
                state = self.env.reset()
