import copy
import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from collections import deque
from tqdm import tqdm

def train(policy, env, replay_buffer, critic1, critic2,
          gamma=0.99, tau=0.01, alpha=0.2, lr=1e-4,
          n_episodes=1000, update_every=10, warmup_steps=1000, batch_size=256,
          checkpoint_path="checkpoints/best_policy_model.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    policy  = policy.to(device)
    critic1 = critic1.to(device)
    critic2 = critic2.to(device)

    # Frozen target critics: stabilize the Bellman target by tracking critic1/critic2
    # with a slow-moving (Polyak-averaged) copy instead of bootstrapping off themselves.
    target1 = copy.deepcopy(critic1).to(device)
    target2 = copy.deepcopy(critic2).to(device)
    for p in target1.parameters():
        p.requires_grad = False
    for p in target2.parameters():
        p.requires_grad = False

    actor_opt = torch.optim.Adam(policy.parameters(),  lr=lr)
    critic_opt = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=lr
    )

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    recent_rewards = deque(maxlen=50)
    best_mean_reward = -np.inf
    total_steps = 0

    episode_bar = tqdm(range(n_episodes), desc="Training hedge policy", unit="episode")
    for ep in episode_bar:
        state = env.reset()
        done  = False
        episode_reward = 0.0

        while not done:
            total_steps += 1

            # --- Act ---
            # Random actions during warmup collect diverse experience before the policy starts driving exploration.
            if total_steps <= warmup_steps:
                action = np.random.uniform(-1, 1)
            else:
                with torch.no_grad():
                    dist   = policy(torch.tensor(state, dtype=torch.float32, device=device))
                    action = dist.sample().clamp(-1, 1).item()

            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if total_steps <= warmup_steps or len(replay_buffer) < batch_size or total_steps % update_every != 0:
                continue

            # --- Sample batch ---
            s, a, r, s2, d = replay_buffer.sample(batch_size)
            s, a, r, s2, d = s.to(device), a.to(device), r.to(device), s2.to(device), d.to(device)

            # --- Critic update (Bellman) ---
            with torch.no_grad():
                dist2      = policy(s2)
                a2         = dist2.rsample()
                log_pi2    = dist2.log_prob(a2).sum(-1, keepdim=True)
                q_target   = r + gamma * (1 - d) * (
                    torch.min(target1(s2, a2), target2(s2, a2)) - alpha * log_pi2
                )

            critic_loss = (
                F.mse_loss(critic1(s, a), q_target) +
                F.mse_loss(critic2(s, a), q_target)
            )

            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

            with torch.no_grad():
                for p, tp in zip(critic1.parameters(), target1.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)
                for p, tp in zip(critic2.parameters(), target2.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)

            # --- Actor update ---
            dist_new  = policy(s)
            a_new     = dist_new.rsample()
            log_pi    = dist_new.log_prob(a_new).sum(-1, keepdim=True)
            actor_loss = (alpha * log_pi - torch.min(
                critic1(s, a_new), critic2(s, a_new)
            )).mean()

            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

        recent_rewards.append(episode_reward)
        mean_reward = np.mean(recent_rewards)
        episode_bar.set_postfix(reward=f"{episode_reward:.4f}", mean_reward_50=f"{mean_reward:.4f}")

        if len(recent_rewards) == recent_rewards.maxlen and mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save(policy.state_dict(), checkpoint_path)
            tqdm.write(f"Saved new best policy (mean reward={mean_reward:.4f})")

def evaluate(action_fn, env, n_episodes=1000):
    """
    action_fn: state (np.ndarray) -> hedge position (float). For a trained RL
    policy, pass e.g. `lambda s: policy(torch.tensor(s, dtype=torch.float32)).mean.item()`
    (the deterministic/mean action); for a baseline, pass one of
    AmericanHedging.baselines directly - they already match this signature.
    """
    pnls = []
    for ep in tqdm(range(n_episodes), desc="Evaluating hedge policy", unit="episode"):
        state, done, total_reward = env.reset(), False, 0
        while not done:
            action = action_fn(state)
            state, reward, done = env.step(action)
            total_reward += reward
        pnls.append(total_reward)

    mean_pnl = np.mean(pnls)
    std_pnl  = np.std(pnls)
    cvar_95  = np.percentile(pnls, 5)

    print(f"Mean P&L:          {mean_pnl:.4f}")
    print(f"Std  P&L (error):  {std_pnl:.4f}")   # your hedging error
    print(f"CVaR 95%:          {cvar_95:.4f}")

    return mean_pnl, std_pnl, cvar_95
