# D1 - Q-learning (reinforcement learning with a table)
#
# RL is different from everything before. There's no dataset with right answers.
# An agent tries actions in an environment, gets rewards, and has to figure out
# by itself which actions lead to more reward later.
#
# Q-learning keeps a table Q[state, action] = "how much total reward do i expect if
# i do this action here and play well after". After every step i update it with
#   Q[s,a] += alpha * (reward + gamma * max Q[next_s] - Q[s,a])
# and to explore, it does a random action with probability epsilon, which
# starts at 1 (all random) and slowly goes down.
#
# 2 games from gymnasium:
#   FrozenLake - 4x4 ice grid, get from S to G without falling in holes (H).
#                the ice is slippery, so you only go where you wanted 1/3 of the time.
#                reward 1 only when you reach the goal
#   Taxi       - pick up a passenger and drop them at the right spot.
#                -1 per step, +20 for a correct drop off, -10 for a wrong pickup/dropoff
# Baseline for both: an agent that just does random actions.
#
# pip install gymnasium
# python d1_qlearning.py

import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)


def q_learning(env, episodes, alpha=0.1, gamma=0.99, eps_min=0.05):
    q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    for ep in range(episodes):
        # epsilon goes from 1 down to eps_min over the first 80% of episodes
        eps = max(eps_min, 1 - ep / (0.8 * episodes))
        s, _ = env.reset(seed=int(rng.integers(1e9)))
        total, done = 0, False
        while not done:
            if rng.random() < eps:
                a = env.action_space.sample()
            else:
                # pick the best action, random among ties (at the start everything is 0)
                a = rng.choice(np.flatnonzero(q[s] == q[s].max()))
            s2, r, terminated, truncated, _ = env.step(a)
            # if the episode really ended there's no future reward
            target = r if terminated else r + gamma * q[s2].max()
            q[s, a] += alpha * (target - q[s, a])
            s, total, done = s2, total + r, terminated or truncated
        rewards.append(total)
    return q, np.array(rewards)


def evaluate(env, policy, episodes=1000):
    totals, steps = [], []
    for _ in range(episodes):
        s, _ = env.reset(seed=int(rng.integers(1e9)))
        total, n, done = 0, 0, False
        while not done:
            s, r, terminated, truncated, _ = env.step(policy(s))
            total, n, done = total + r, n + 1, terminated or truncated
        totals.append(total)
        steps.append(n)
    return np.mean(totals), np.mean(steps)


def show_frozenlake_policy(env, q):
    arrows = "←↓→↑"   # FrozenLake actions: 0 left, 1 down, 2 right, 3 up
    desc = env.unwrapped.desc.astype(str)
    print("\nwhat it learned to do on each square (H = hole, G = goal):")
    for i in range(desc.shape[0]):
        row = []
        for j in range(desc.shape[1]):
            c = desc[i, j]
            row.append(c if c in "HG" else arrows[q[i * desc.shape[1] + j].argmax()])
        print("   " + " ".join(row))


def make_taxi(**kw):
    # newer gymnasium renamed Taxi-v3 to Taxi-v4, older ones only have v3
    try:
        return gym.make("Taxi-v4", **kw)
    except gym.error.Error:
        return gym.make("Taxi-v3", **kw)


def rolling(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ---- FrozenLake ----
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    t0 = time.time()
    q, rewards = q_learning(env, episodes=20000)
    print(f"FrozenLake: trained for 20,000 episodes ({time.time() - t0:.0f}s)")
    rand_win, rand_steps = evaluate(env, lambda s: env.action_space.sample())
    q_win, q_steps = evaluate(env, lambda s: int(q[s].argmax()))
    print(f"  random agent reaches the goal {rand_win:6.1%} of the time  (avg {rand_steps:.0f} steps)")
    print(f"  q-learning   reaches the goal {q_win:6.1%} of the time  (avg {q_steps:.0f} steps)")
    show_frozenlake_policy(env, q)
    axes[0].plot(rolling(rewards, 500), label="q-learning (while exploring)")
    axes[0].axhline(rand_win, color="gray", ls="--", label="random agent")
    axes[0].axhline(q_win, color="green", ls=":", label="final greedy policy")
    axes[0].set(title="FrozenLake 4x4 (slippery)", xlabel="episode", ylabel="success rate (last 500)")
    axes[0].legend()

    # ---- Taxi ----
    env = make_taxi()
    t0 = time.time()
    q, rewards = q_learning(env, episodes=5000)
    print(f"\nTaxi: trained for 5,000 episodes ({time.time() - t0:.0f}s), "
          f"table size {q.shape[0]} states x {q.shape[1]} actions")
    rand_r, rand_steps = evaluate(env, lambda s: env.action_space.sample(), episodes=200)
    q_r, q_steps = evaluate(env, lambda s: int(q[s].argmax()))
    print(f"  random agent: avg reward {rand_r:7.1f}  (avg {rand_steps:.0f} steps, 200 = gave up)")
    print(f"  q-learning:   avg reward {q_r:7.1f}  (avg {q_steps:.0f} steps)")
    axes[1].plot(rolling(rewards, 100), label="q-learning (while exploring)")
    axes[1].axhline(rand_r, color="gray", ls="--", label="random agent")
    axes[1].axhline(q_r, color="green", ls=":", label="final greedy policy")
    axes[1].set(title="Taxi", xlabel="episode", ylabel="reward per episode (last 100)")
    axes[1].legend()

    # watch one taxi ride with the learned policy
    env = make_taxi()
    s, _ = env.reset(seed=1)
    names = ["down", "up", "right", "left", "pickup", "dropoff"]
    path, done, total = [], False, 0
    while not done and len(path) < 50:
        a = int(q[s].argmax())
        path.append(names[a])
        s, r, terminated, truncated, _ = env.step(a)
        total, done = total + r, terminated or truncated
    print(f"\none taxi ride with the learned policy ({len(path)} steps, reward {total}):")
    print("  " + " -> ".join(path))

    fig.suptitle("D1 - Q-learning vs random agent")
    fig.tight_layout()
    fig.savefig("d1_qlearning.png", dpi=120)
    print("\nsaved d1_qlearning.png")


if __name__ == "__main__":
    main()
