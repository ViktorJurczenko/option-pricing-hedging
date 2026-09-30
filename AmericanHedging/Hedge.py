import copy
import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from collections import deque
from tqdm import tqdm

def _squash(dist: torch.distributions.Distribution, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterized sample from `dist`, squashed into (-1, 1) via tanh,
    with the log-prob Jacobian correction for that squashing.

    A hard `.clamp(-1, 1)` has exactly zero gradient once a sample lands
    outside the range, which both freezes `mean` wherever it happens to have
    drifted to and, worse, gives the entropy term a "free" way to keep
    growing `std` (log_prob at a fixed clamped point under an ever-wider
    distribution keeps shrinking, with no offsetting cost once the clamped
    action itself no longer changes) - std saturates and mean never comes
    back. tanh's gradient is never exactly zero, and the correction term
    below (`log(1 - tanh(x)^2)`) grows with |x|, so pushing the pre-squash
    sample further out has a real, increasing entropy cost instead of a free
    one. This is the standard SAC fix for bounded continuous actions.
    """
    raw     = dist.rsample()
    squashed = torch.tanh(raw)
    log_prob = (
        dist.log_prob(raw).sum(-1, keepdim=True)
        - torch.log(1 - squashed.pow(2) + eps).sum(-1, keepdim=True)
    )
    return squashed, log_prob

def train(policy, env, replay_buffer, critic1, critic2,
          gamma=0.99, tau=0.01, alpha=0.2, lr=1e-4,
          n_episodes=1000, update_every=10, warmup_steps=1000, batch_size=256,
          grad_steps_per_update=1, risk_lambda=0.0, reward_scale=1.0,
          auto_alpha=False, target_entropy=-1.0, alpha_lr=None,
          checkpoint_path="checkpoints/best_policy_model.pth"):
    """
    reward_scale: multiplies the env reward before the risk penalty, i.e. the
        critics/actor train on `x - risk_lambda * x**2` with x = reward_scale *
        reward. Hedging rewards are ~1e-2 per step, orders of magnitude below
        the entropy term (alpha * log_pi ~ O(0.1)), which otherwise dominates
        the actor loss.
    auto_alpha: learn the entropy coefficient (standard SAC automatic entropy
        tuning) instead of keeping it fixed; `alpha` is then only its initial
        value, and it's adjusted to keep the policy's entropy near
        `target_entropy` (conventionally -action_dim).
    alpha_lr: Adam learning rate for log_alpha when auto_alpha is set; None
        (default) reuses `lr`. Only log_alpha's own step size - independent
        of the actor/critic rate, since it controls how fast exploration
        adapts rather than how fast the networks fit.
    """
    if grad_steps_per_update < 1:
        # 0 (or negative) would silently skip every critic/actor update below,
        # running a full n_episodes training loop that never actually learns.
        raise ValueError(f"grad_steps_per_update must be >= 1, got {grad_steps_per_update}")
    if alpha_lr is not None and alpha_lr <= 0:
        raise ValueError(f"alpha_lr must be > 0, got {alpha_lr}")
    if reward_scale <= 0:
        raise ValueError(f"reward_scale must be > 0, got {reward_scale}")
    if alpha <= 0 and auto_alpha:
        # log(alpha) is the learned parameter.
        raise ValueError(f"alpha must be > 0 when auto_alpha is set, got {alpha}")

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
    if auto_alpha:
        log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=device, requires_grad=True)
        alpha_opt = torch.optim.Adam([log_alpha], lr=lr if alpha_lr is None else alpha_lr)

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
            # `action` is the squashed policy output the critics train on; the env
            # gets policy.hedge_ratio of it (delta + a correction for a residual policy).
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            # Random actions during warmup collect diverse experience before the policy starts driving exploration.
            if total_steps <= warmup_steps:
                action = np.random.uniform(-1, 1)
            else:
                with torch.no_grad():
                    # Not routed through _squash: no gradient is taken here (no_grad,
                    # single scalar action for the env), so .sample() over .rsample()
                    # and skipping the unused log-prob term are both fine.
                    action = torch.tanh(policy(state_t).sample()).item()
            with torch.no_grad():
                hedge = policy.hedge_ratio(
                    state_t, torch.tensor([action], dtype=torch.float32, device=device)
                ).item()

            next_state, reward, done = env.step(hedge)
            # Critic/actor train on a scaled, risk-shaped reward (mean-variance
            # style); episode_reward (and thus the checkpoint criterion) stays
            # on the raw env reward so it remains comparable to test-hedge's P&L.
            scaled_reward = reward_scale * reward
            shaped_reward = scaled_reward - risk_lambda * scaled_reward ** 2
            replay_buffer.push(state, action, shaped_reward, next_state, done)
            state = next_state
            episode_reward += reward

            if total_steps <= warmup_steps or len(replay_buffer) < batch_size or total_steps % update_every != 0:
                continue

            for _ in range(grad_steps_per_update):
                # --- Sample batch ---
                s, a, r, s2, d = replay_buffer.sample(batch_size)
                s, a, r, s2, d = s.to(device), a.to(device), r.to(device), s2.to(device), d.to(device)

                # --- Critic update (Bellman) ---
                with torch.no_grad():
                    dist2      = policy(s2)
                    a2, log_pi2 = _squash(dist2)
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
                a_new, log_pi = _squash(dist_new)
                actor_loss = (alpha * log_pi - torch.min(
                    critic1(s, a_new), critic2(s, a_new)
                )).mean()

                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                if auto_alpha:
                    alpha_loss = -(log_alpha * (log_pi.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
                    alpha = log_alpha.exp().item()

        recent_rewards.append(episode_reward)
        mean_reward = np.mean(recent_rewards)
        postfix = {"reward": f"{episode_reward:.4f}", "mean_reward_50": f"{mean_reward:.4f}"}
        if auto_alpha:
            postfix["alpha"] = f"{alpha:.4f}"
        episode_bar.set_postfix(**postfix)

        if len(recent_rewards) == recent_rewards.maxlen and mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save(policy.state_dict(), checkpoint_path)
            tqdm.write(f"Saved new best policy (mean reward={mean_reward:.4f})")

def evaluate(action_fn, env, n_episodes=1000):
    """
    action_fn: state (np.ndarray) -> hedge position (float). For a trained RL
    policy, map the deterministic action (tanh-squashed distribution mean,
    matching how train() squashes sampled actions - see `_squash`) through
    `policy.hedge_ratio`, as scripts/test_hedge.py does; for a baseline, pass
    one of AmericanHedging.baselines directly - they already match this
    signature.
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
