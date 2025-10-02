import random

from matplotlib.colors import LogNorm

from rl.env import Labyrinth
from rl.qlearning import QLearning
from rl.value_iteration import ValueIteration
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray

EPISODES = 1000
RUNS = 20
MAX_STEPS = 200  # cap par épisode pour éviter les boucles


def plot_values(values: NDArray[np.float64]) -> None:
    """
    Plots a heatmap representing the state values in a grid world.

    Parameters:
    - values (NDArray[np.float64]): A 2D numpy array of shape (height, width) where each element
                                    represents the computed value of that state.

    Returns:
    - None: Displays the plot.
    """
    assert values.ndim == 2, f"Expected 2D array of shape (height, width), got shape {values.shape}"
    sns.heatmap(values, annot=True, cbar_kws={'label': 'Value'})
    plt.savefig("values_heatmap.png")


def plot_qvalues(q_values: NDArray[np.float64], qType: int, action_symbols: list[str]) -> None:
    """
    Plots a heatmap of the maximum Q-values in each state of a grid world and overlays symbols
    to represent the optimal action in each state.

    Parameters:
    - q_values (NDArray[np.float64]): A 3D numpy array of shape (height, width, n_actions), where each cell contains Q-values
                                      for four possible actions (up, down, right, left).
    - env (Labyrinth): The environment instance to access action symbols.

    Returns:
    - None: Displays the plot.
    """
    assert q_values.ndim == 3, f"Expected 3D array of shape (height, width, n_actions), got shape {q_values.shape}"
    assert q_values.shape[-1] == len(action_symbols), f"Number of action symbols should match the number of actions"
    height, width = q_values.shape[:2]

    # Calculate the best action and max Q-value for each cell
    best_actions = np.argmax(q_values, axis=2)
    max_q_values = np.max(q_values, axis=2)

    # Plotting the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(max_q_values, origin="upper")
    plt.colorbar(label="Max Q-value")
    # Overlay best action symbols
    for i in range(height):
        for j in range(width):
            action_symbol = action_symbols[best_actions[i, j]]
            plt.text(j, i, action_symbol, ha='center', va='center', color='black', fontsize=12)

    # Labels and layout
    if qType == 0:
        plt.title("Q-value Heatmap with Optimal Actions (ε-greedy)")
        plt.grid(False)
        plt.savefig("qvalues_heatmap_egreedy.png")
    elif qType == 1:
        plt.title("Q-value Heatmap with Optimal Actions (Softmax)")
        plt.grid(False)
        plt.savefig("qvalues_heatmap_softmax.png")


def random_moves(env: Labyrinth, n_steps: int) -> None:
    """
    Makes random moves in the environment and renders each step.

    Parameters:
    - env (Labyrinth): The environment instance where random moves will be performed.
    - n_steps (int): Number of random steps to perform.

    Returns:
    - None
    """
    env.reset()
    env.render()
    episode_rewards = 0
    for s in range(n_steps):

        random_action = np.random.choice(env.get_all_actions())
        reward = env.step(random_action)
        done = env.is_done()
        episode_rewards += reward
        if done:
            print("collected reward =", episode_rewards)
            env.reset()
            episode_rewards = 0
        env.render()


def set_seed(s):
    np.random.seed(s);
    random.seed(s)


# plannings d'exploration
def const_eps(eps):  # τ=0
    return lambda ep: (eps, 0.0)


def exp_eps():  # ε: 1 → 0.01
    return lambda ep: (0.01 * (1 / 0.01) ** (1 - ep / max(EPISODES - 1, 1)), 0.0)


def const_tau(tau):  # ε=0
    return lambda ep: (0.0, tau)


def exp_tau():  # τ: 100 → 0.01
    return lambda ep: (0.0, 0.01 * (100 / 0.01) ** (1 - ep / max(EPISODES - 1, 1)))


def run_one(env_p, agent_p, schedule):
    """Un entraînement → retourne (scores[EPISODES], visits[H,W])"""
    env = Labyrinth(**env_p)
    H, W = env.get_map_size()
    visits = np.zeros((H, W), dtype=np.int64)
    agent = QLearning(env, **agent_p)

    scores = np.zeros(EPISODES, dtype=float)
    for ep in range(EPISODES):
        eps, tau = schedule(ep)
        agent.epsilon, agent.tau = eps, tau

        s = env.reset()
        G = 0.0
        for t in range(MAX_STEPS):
            visits[s] += 1
            a = agent.choose_action(s)
            r = env.step(a)
            s2 = env.get_observation()
            done = env.is_done()
            agent.update(s, a, r, s2, done)  # IMPORTANT: gère done
            G += r
            s = s2
            if done: break
        scores[ep] = G
    return scores, visits


def average_runs(configs, title_curve, prefix_heat):
    """Lance RUNS seeds, trace la courbe moyenne + sauvegarde heatmaps."""
    all_curves = []  # (label, mean)
    all_visits = []  # visites cumulées (pour normaliser l'échelle)
    for label, schedule in configs:
        runs_scores = []
        runs_visits = []
        for seed in range(RUNS):
            set_seed(seed)
            env_p = dict(malfunction_probability=0.0)  # p=0 pour cette section
            agent_p = dict(gamma=0.95, alpha=0.1, epsilon=0.0, tau=0.0)
            sc, vis = run_one(env_p, agent_p, schedule)
            runs_scores.append(sc)
            runs_visits.append(vis)
        mean_curve = np.mean(np.stack(runs_scores), axis=0)
        all_curves.append((label, mean_curve))
        all_visits.append(np.sum(np.stack(runs_visits), axis=0))  # somme sur RUNS
        # sauvegarde heatmap plus tard avec vmax commun
    # trace le graphe (les ylim seront fixées à l'extérieur)
    x = np.arange(EPISODES)
    for label, mean_curve in all_curves:
        plt.plot(x, mean_curve, label=label)
    plt.xlabel("Épisode");
    plt.ylabel("Score moyen par épisode");
    plt.title(title_curve);
    plt.legend();
    plt.tight_layout()
    plt.savefig(f"{prefix_heat}_mean_return.png", dpi=150)
    print(f"[saved] {prefix_heat}_mean_return.png")
    return all_curves, all_visits


def save_visits_heatmaps(configs, visits_list, prefix):
    # vmax global commun à toutes les heatmaps
    vmax_global = max((V + 1).max() for V in visits_list)
    for (label, _), V in zip(configs, visits_list):
        plt.figure()
        plt.imshow(V + 1, norm=LogNorm(vmin=1, vmax=vmax_global))
        plt.colorbar(label="Visites (log)")
        plt.title(f"Visites – {label}")
        plt.xticks(range(V.shape[1]))
        plt.yticks(range(V.shape[0]))
        plt.tight_layout()
        fname = f"{prefix}_visits_{label.replace(' ', '_').replace('→', 'to')}.png"
        plt.savefig(fname, dpi=150);
        print(f"[saved] {fname}")


def effet_exploration():
    # 1) ε-greedy (τ=0)
    eps_cfgs = [
        ("ε=0.2", const_eps(0.2)),
        ("ε=0.1", const_eps(0.1)),
        ("ε=0.01", const_eps(0.01)),
        ("ε: exp 1→0.01", exp_eps()),
    ]
    # 2) Max-Boltzmann (ε=0)
    tau_cfgs = [
        ("τ=0.01", const_tau(0.01)),
        ("τ=1", const_tau(1.0)),
        ("τ=10", const_tau(10.0)),
        ("τ=100", const_tau(100.0)),
        ("τ: exp 100→0.01", exp_tau()),
    ]

    # Lancer et tracer
    plt.figure()
    eps_curves, eps_visits = average_runs(eps_cfgs, "ε-greedy : score moyen par épisode", "eps")
    y1 = plt.gca().get_ylim();
    plt.close()

    plt.figure()
    tau_curves, tau_visits = average_runs(tau_cfgs, "Max-Boltzmann : score moyen par épisode", "tau")
    y2 = plt.gca().get_ylim();
    plt.close()

    # Heatmaps (log) avec vmax commun
    save_visits_heatmaps(eps_cfgs, eps_visits, "eps")
    save_visits_heatmaps(tau_cfgs, tau_visits, "tau")


def run_value_iteration_deltas():
    env = Labyrinth()
    deltas = [1, 0.1, 0.01, 0.001]
    value_tables = {}

    for delta in deltas:
        vi = ValueIteration(env, gamma=0.9)
        vi.train(delta=delta, max_iterations=1000)
        value_tables[delta] = vi.get_value_table()

        # Plot heatmap of values
        plt.figure(figsize=(6, 6))
        plt.imshow(value_tables[delta], cmap="viridis")
        plt.colorbar(label="State Value")
        plt.title(f"Value Iteration Heatmap (δ={delta})")
        plt.savefig(f"value_heatmap_delta_{str(delta).replace('.', '')}.png")
        plt.close()

    return value_tables



if __name__ == "__main__":
    env = Labyrinth(malfunction_probability=0.0)
    #run_value_iteration_deltas()
    vi = ValueIteration(env=env)
    vi.train(delta=0.01)
    plot_values(vi.get_value_table())
    ql_eps = QLearning(env=env, alpha=0.1, epsilon=0.1)
    ql_eps.train(10_000)
    plot_qvalues(ql_eps.get_q_table(), 0, action_symbols=Labyrinth.ACTION_SYMBOLS)

    ql_soft = QLearning(env=env, alpha=0.1, tau=100)
    ql_soft.train(10_000)
    plot_qvalues(ql_soft.get_q_table(), 1, action_symbols=Labyrinth.ACTION_SYMBOLS)

    effet_exploration()
