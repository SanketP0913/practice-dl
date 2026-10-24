# D2 - DQN on CartPole (Q-learning with a neural network)
#
# In D1 the Q-table worked because FrozenLake has 16 states and Taxi 500.
# CartPole has 4 continuous numbers (cart position, cart speed, pole angle,
# pole angular speed), so there are endless states and a table doesn't work directly.
# DQN replaces the table with a small neural network: 4 numbers in -> Q value for
# "push left" and "push right" out.
#
# The game: balance a pole on a cart by pushing left or right. +1 reward for every
# step the pole stays up, max 500 steps.
#
# What i compare:
#   random            - random pushes (baseline)
#   my rule           - push towards the side the pole is falling to (hand written)
#   q-table           - D1's q-learning, after chopping the 4 numbers into bins
#   dqn               - neural network + 2 tricks that make it stable:
#                         replay buffer: remember old steps and learn from random batches of them
#                         target network: a frozen copy of the net for the targets, updated slowly
#
# small network so it runs on the cpu, it's faster than mps for this.
# pip install gymnasium torch
# python d2_dqn_cartpole.py

import copy
import random
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def evaluate(env, policy, episodes=100):
    scores = []
    for i in range(episodes):
        s, _ = env.reset(seed=10_000 + i)
        total, done = 0, False
        while not done:
            s, r, terminated, truncated, _ = env.step(policy(s))
            total, done = total + r, terminated or truncated
        scores.append(total)
    return np.mean(scores)


def my_rule(s):
    # s = [cart pos, cart speed, pole angle, pole angular speed]
    # if the pole leans (or is falling) right, push right, otherwise push left
    return 1 if s[2] + 0.5 * s[3] > 0 else 0


# ---------- q-table with bins ----------
BINS = [np.linspace(-2.4, 2.4, 5), np.linspace(-3, 3, 5),
        np.linspace(-0.21, 0.21, 9), np.linspace(-3.5, 3.5, 9)]


def to_bins(s):
    # turn the 4 numbers into one table index
    idx = 0
    for x, b in zip(s, BINS):
        idx = idx * (len(b) + 1) + int(np.digitize(x, b))
    return idx


def q_table(env, episodes=3000, alpha=0.1, gamma=0.99):
    n_states = int(np.prod([len(b) + 1 for b in BINS]))
    q = np.zeros((n_states, 2))
    scores = []
    for ep in range(episodes):
        eps = max(0.05, 1 - ep / (0.8 * episodes))
        s, _ = env.reset(seed=ep)
        si, total, done = to_bins(s), 0, False
        while not done:
            a = env.action_space.sample() if random.random() < eps else int(q[si].argmax())
            s2, r, terminated, truncated, _ = env.step(a)
            s2i = to_bins(s2)
            target = r if terminated else r + gamma * q[s2i].max()
            q[si, a] += alpha * (target - q[si, a])
            si, total, done = s2i, total + r, terminated or truncated
        scores.append(total)
    return q, scores


# ---------- dqn ----------
def make_net():
    return nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))


# first try was eps_steps=10_000, tau=0.005 and it got stuck around 100-300 steps.
# exploring for a shorter time + a faster target net got to 500 on 2 different seeds
def dqn(env, episodes=400, gamma=0.99, lr=5e-4, batch=64, buffer_size=50_000,
        warmup=1000, eps_steps=5_000, tau=0.01):
    net = make_net()
    target = copy.deepcopy(net)     # the frozen copy
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buffer = deque(maxlen=buffer_size)
    scores, evals = [], []
    best_score, best_net = -1, None
    step = 0
    greedy = lambda s: int(net(torch.tensor(s, dtype=torch.float32)).argmax())

    for ep in range(episodes):
        s, _ = env.reset(seed=ep)
        total, done = 0, False
        while not done:
            eps = max(0.05, 1 - step / eps_steps)
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = greedy(s)
            s2, r, terminated, truncated, _ = env.step(a)
            buffer.append((s, a, r, s2, float(terminated)))
            s, total, done = s2, total + r, terminated or truncated
            step += 1

            if len(buffer) >= warmup:
                # learn from a random batch of old steps, not just the last one
                b = random.sample(buffer, batch)
                st, ac, rw, st2, term = (torch.tensor(np.array(x), dtype=torch.float32) for x in zip(*b))
                q = net(st).gather(1, ac.long().unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target(st2).max(1).values
                    y = rw + gamma * q_next * (1 - term)
                loss = F.smooth_l1_loss(q, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10)
                opt.step()
                # move the target net a tiny bit towards the real net
                with torch.no_grad():
                    for p, tp in zip(net.parameters(), target.parameters()):
                        tp.mul_(1 - tau).add_(tau * p)
        scores.append(total)

        # dqn can suddenly get worse while training, so every 20 episodes i test
        # the greedy policy and keep the best version
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                score = evaluate(env, greedy, episodes=10)
            evals.append((ep + 1, score))
            mark = ""
            if score >= best_score:
                best_score, best_net, mark = score, copy.deepcopy(net), "  <- best"
            print(f"  episode {ep + 1:3d}  last 20 avg {np.mean(scores[-20:]):6.1f}  "
                  f"greedy test {score:6.1f}  eps {eps:.2f}{mark}")
    return best_net, scores, evals


def main():
    env = gym.make("CartPole-v1")
    results = {}

    results["random"] = evaluate(env, lambda s: env.action_space.sample())
    results["my rule"] = evaluate(env, my_rule)
    print(f"random pushes: {results['random']:.1f} steps   my rule: {results['my rule']:.1f} steps")

    t0 = time.time()
    q, q_scores = q_table(env)
    results["q-table (bins)"] = evaluate(env, lambda s: int(q[to_bins(s)].argmax()))
    print(f"q-table: trained 3000 episodes in {time.time() - t0:.0f}s, "
          f"table has {q.shape[0]:,} states -> {results['q-table (bins)']:.1f} steps")

    print("\ntraining dqn:")
    t0 = time.time()
    net, d_scores, evals = dqn(env)
    with torch.no_grad():
        results["dqn"] = evaluate(env, lambda s: int(net(torch.tensor(s, dtype=torch.float32)).argmax()))
    print(f"dqn trained in {time.time() - t0:.0f}s")

    print("\naverage steps balanced over 100 test episodes (max 500):")
    for k, v in results.items():
        print(f"  {k:16s} {v:6.1f}")

    torch.save(net.state_dict(), "d2_dqn.pt")
    fig, ax = plt.subplots(figsize=(10, 5))
    roll = lambda x: np.convolve(x, np.ones(20) / 20, mode="valid")
    ax.plot(roll(q_scores), label="q-table while training", alpha=0.7)
    ax.plot(roll(d_scores), label="dqn while training", alpha=0.7)
    ax.plot(*zip(*evals), "o-", label="dqn greedy test")
    for k, c in [("random", "gray"), ("my rule", "orange")]:
        ax.axhline(results[k], color=c, ls="--", label=k)
    ax.set(xlabel="episode", ylabel="steps balanced (20 episode avg)", title="D2 - CartPole", ylim=(0, 520))
    ax.legend()
    fig.tight_layout()
    fig.savefig("d2_cartpole.png", dpi=120)
    print("\nsaved d2_cartpole.png and d2_dqn.pt")


if __name__ == "__main__":
    main()
