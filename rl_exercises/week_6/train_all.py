"""
Trains ActorCriticAgent with all four baseline types on CartPole-v1 and LunarLander-v3.
Saves evaluation results to results/results.json for later plotting with RLiable.

Usage:
    python train_all.py
"""

from typing import Dict, List

import json
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Adjust these imports to match your project layout
# ---------------------------------------------------------------------------
from rl_exercises.week_6.networks import Policy, ValueNetwork
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Minimal self-contained ActorCriticAgent (no hydra / AbstractAgent needed)
# ---------------------------------------------------------------------------


def set_seed(env: gym.Env, seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class ActorCriticAgent:
    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 5e-3,
        lr_critic: float = 1e-2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        seed: int = 0,
        hidden_size: int = 128,
        baseline_type: str = "value",
        baseline_decay: float = 0.9,
    ) -> None:
        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.baseline_type = baseline_type
        self.baseline_decay = baseline_decay

        self.policy = Policy(env.observation_space, env.action_space, hidden_size)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)

        if baseline_type in ("value", "gae"):
            self.value_fn = ValueNetwork(env.observation_space, hidden_size)
            self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=lr_critic)

        if baseline_type == "avg":
            self.running_return = 0.0

    def predict_action(self, state: np.ndarray, evaluate: bool = False):
        t = torch.from_numpy(state).float()
        probs = self.policy(t).squeeze(0)
        if evaluate:
            return int(torch.argmax(probs).item()), None
        dist = Categorical(probs)
        action = dist.sample().item()
        return action, dist.log_prob(torch.tensor(action))

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def compute_advantages(self, states, rewards):
        ret = self.compute_returns(rewards)
        state_batch = torch.stack([torch.from_numpy(s).float() for s in states])
        values = self.value_fn(state_batch).squeeze(-1)
        adv = ret - values.detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return adv, ret

    def compute_gae(self, states, rewards, next_states, dones):
        state_batch = torch.stack([torch.from_numpy(s).float() for s in states])
        next_state_batch = torch.stack(
            [torch.from_numpy(s).float() for s in next_states]
        )
        with torch.no_grad():
            values = self.value_fn(state_batch).squeeze(-1)
            next_values = self.value_fn(next_state_batch).squeeze(-1)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        deltas = rewards_t + self.gamma * next_values * (1.0 - dones_t) - values
        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            gae = (
                deltas[t].item()
                + self.gamma * self.gae_lambda * (1.0 - dones_t[t].item()) * gae
            )
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        return advantages.detach(), returns.detach()

    def update_agent(self, trajectory):
        states, actions, rewards, next_states, dones, log_probs = zip(*trajectory)

        if self.baseline_type == "gae":
            adv, ret = self.compute_gae(
                list(states), list(rewards), list(next_states), list(dones)
            )
        elif self.baseline_type == "value":
            adv, ret = self.compute_advantages(list(states), list(rewards))
        elif self.baseline_type == "avg":
            ret = self.compute_returns(list(rewards))
            adv = ret - self.running_return
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            self.running_return = (
                self.baseline_decay * self.running_return
                + (1.0 - self.baseline_decay) * ret.mean().item()
            )
        else:  # none
            ret = self.compute_returns(list(rewards))
            adv = (ret - ret.mean()) / (ret.std(unbiased=False) + 1e-8)

        logp_t = torch.stack(log_probs)
        policy_loss = -(logp_t * adv).mean()
        self.policy_optimizer.zero_grad()
        if policy_loss.requires_grad:
            policy_loss.backward()
            self.policy_optimizer.step()

        if self.baseline_type in ("value", "gae"):
            vals = self.value_fn(
                torch.stack([torch.from_numpy(s).float() for s in states])
            ).squeeze(-1)
            value_loss = F.mse_loss(vals, ret)
            self.value_optimizer.zero_grad()
            if value_loss.requires_grad:
                value_loss.backward()
                self.value_optimizer.step()

    def evaluate(self, eval_env: gym.Env, num_episodes: int = 5) -> float:
        self.policy.eval()
        returns = []
        with torch.no_grad():
            for _ in range(num_episodes):
                state, _ = eval_env.reset()
                done = False
                total_r = 0.0
                while not done:
                    action, _ = self.predict_action(state, evaluate=True)
                    state, r, term, trunc, _ = eval_env.step(action)
                    done = term or trunc
                    total_r += r
                returns.append(total_r)
        self.policy.train()
        return float(np.mean(returns))

    def train(
        self,
        total_steps: int = 200_000,
        eval_interval: int = 10_000,
        eval_episodes: int = 5,
    ) -> Dict[str, List]:
        """
        Returns a dict:
            {"steps": [...], "returns": [...]}
        where each entry in "returns" is the mean eval return at that step.
        """
        eval_env = gym.make(self.env.spec.id)
        step_count = 0
        log = {"steps": [], "returns": []}

        while step_count < total_steps:
            state, _ = self.env.reset()
            done = False
            trajectory = []

            while not done and step_count < total_steps:
                action, logp = self.predict_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                trajectory.append(
                    (state, action, float(reward), next_state, done, logp)
                )
                state = next_state
                step_count += 1

                if step_count % eval_interval == 0:
                    mean_r = self.evaluate(eval_env, num_episodes=eval_episodes)
                    log["steps"].append(step_count)
                    log["returns"].append(mean_r)
                    print(f"  step {step_count:7d}  return {mean_r:8.1f}")

            self.update_agent(trajectory)

        eval_env.close()
        return log


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

ENVS = ["CartPole-v1", "LunarLander-v3"]
BASELINES = ["none", "avg", "value", "gae"]
NUM_SEEDS = 3  # run each combo with 3 seeds for confidence intervals
TOTAL_STEPS = 200_000
EVAL_INTERVAL = 10_000
EVAL_EPISODES = 5

HYPERPARAMS = {
    "CartPole-v1": dict(
        lr_actor=5e-3,
        lr_critic=1e-2,
        gamma=0.99,
        gae_lambda=0.95,
        hidden_size=128,
        baseline_decay=0.9,
    ),
    "LunarLander-v3": dict(
        lr_actor=5e-3,
        lr_critic=1e-2,
        gamma=0.99,
        gae_lambda=0.95,
        hidden_size=128,
        baseline_decay=0.9,
    ),
}


def run_all() -> Dict:
    """
    Structure of results dict:
    {
        "CartPole-v1": {
            "none":  {"steps": [...], "returns": [[seed0], [seed1], [seed2]]},
            "avg":   {...},
            ...
        },
        "LunarLander-v3": { ... }
    }
    """
    results = {}

    for env_name in ENVS:
        results[env_name] = {}
        hp = HYPERPARAMS[env_name]

        for baseline in BASELINES:
            print(f"\n{'=' * 60}")
            print(f"  {env_name}  |  baseline={baseline}")
            print(f"{'=' * 60}")

            seed_returns = []
            seed_steps = None

            for seed in range(NUM_SEEDS):
                print(f"\n  -- seed {seed} --")
                env = gym.make(env_name)
                agent = ActorCriticAgent(
                    env,
                    baseline_type=baseline,
                    seed=seed,
                    **hp,
                )
                log = agent.train(
                    total_steps=TOTAL_STEPS,
                    eval_interval=EVAL_INTERVAL,
                    eval_episodes=EVAL_EPISODES,
                )
                env.close()

                seed_returns.append(log["returns"])
                if seed_steps is None:
                    seed_steps = log["steps"]

            results[env_name][baseline] = {
                "steps": seed_steps,
                "returns": seed_returns,  # shape: (num_seeds, num_evals)
            }

    return results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    results = run_all()

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone! Results saved to results/results.json")
