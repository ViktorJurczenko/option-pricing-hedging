"""Terminal P&L of a hedged short American put, the one place the hedging accounting lives.

The hedger sells the put for `p_0` at `t_0`, trades shares on its hedge dates and settles at the exercise day
`τ`. Everything is booked in dollars of `t_0`, discounted by the bank account `e^{r t}`, so cash earning `r`
drops out of the formula. With `S̃_t = e^{-r t} S_t`, a self-financing hedge ends with

    P&L = p_0 + Σ_j δ_j (S̃_{t_{j+1}} - S̃_{t_j}) - Σ_j e^{-r t_j} c_j - e^{-r τ} Z_τ

where `δ_j` is the position chosen at hedge date `t_j` (it earns from `t_j` on), `c_j` the cost of the trade
`δ_j - δ_{j-1}` (with `δ_{-1} = 0`), and `Z_τ` the settlement cost (payoff plus closing trade, `settlement.py`).
The last position earns until `τ`, so on a grid coarser than daily the last interval is cut short at `τ`.

Training, evaluation and every baseline call `hedge_pnl`, so the accounting exists once. The legacy env booked
it twice and both copies shared the bugs (BUGS, B-037 and B-038).

A policy is any callable `policy(inputs, position) -> new_position`. `inputs` is the
`(nb_scenarios, nb_features)` slice of the caller's features at the hedge date, and `position` the
`(nb_scenarios,)` shares held before the trade. What the features are is the caller's choice. This function
only slices them, so a network, a delta hedge and no hedge all fit.
"""

import numbers
from collections.abc import Callable
from dataclasses import dataclass

import torch

from optionpricing.hedging.costs import transaction_cost
from optionpricing.hedging.settlement import cash_settlement, physical_settlement
from optionpricing.hedging.simulator import DAYS_PER_YEAR

SETTLEMENTS = {"cash": cash_settlement, "physical": physical_settlement}

Policy = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class HedgeOutcome:
    """What one batch of hedged scenarios ended with, all in dollars of `t_0`.

    Attributes:
        pnl: terminal P&L of each scenario, shape `(nb_scenarios,)`, differentiable in the policy's parameters.
        costs: discounted transaction costs of each scenario, trades and the settlement's closing trade, shape
            `(nb_scenarios,)`.
        positions: shares held after each hedge date, shape `(nb_scenarios, n_dates)`. A scenario keeps its last
            position after its `τ`, so its final column is the position settled.
    """

    pnl: torch.Tensor
    costs: torch.Tensor
    positions: torch.Tensor


def _as_float(x, dtype=None) -> torch.Tensor:
    """Return `x` as a floating tensor (float64 unless a dtype is given), keeping a tensor's graph."""
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=dtype or torch.float64)
    if not x.is_floating_point():
        x = x.to(dtype or torch.float64)
    return x if dtype is None else x.to(dtype)


def hedge_pnl(
    policy: Policy,
    inputs,
    spot,
    tau,
    premium,
    strike,
    rate,
    rebalance_every: int = 1,
    settlement: str = "cash",
    proportional: float = 0.0,
    fixed: float = 0.0,
    kappa: float | None = None,
) -> HedgeOutcome:
    """Hedge a batch of short puts with `policy` and book each scenario's terminal P&L, discounted to `t_0`.

    Days are on the scenario bank's daily grid (day `d` is `d / DAYS_PER_YEAR` years). The hedge dates are days
    `0, m, 2m, ...` before `τ`, with `m = rebalance_every`. The policy is called once per hedge date on every
    scenario, in order, so a position can depend on the one before it. A scenario whose `τ` has passed keeps its
    position and trades nothing, and its features there are replaced by zeros before the policy sees them: past
    expiry, prices and features are NaN, and a NaN fed to the network would turn its gradients into NaN even
    where the output is discarded. A NaN at or before `τ` is not replaced and makes the P&L NaN.

    Args:
        policy: `policy(inputs, position) -> position`, see the module docstring. It must return a tensor of
            shape `(nb_scenarios,)`.
        inputs: the policy's features, shape `(nb_scenarios, nb_features, n_days)`, time last as `spot`. Built
            by the caller.
        spot: share prices, shape `(nb_scenarios, n_days)`, positive up to each scenario's `τ`. Later days may
            be NaN.
        tau: exercise or expiry day of each scenario, ints in `[0, n_days - 1]`, shape `(nb_scenarios,)`. At
            `τ = 0` the put is settled at once, with no hedge.
        premium: price received for each put at `t_0`, shape `(nb_scenarios,)` or a scalar.
        strike: strike, a scalar or shape `(nb_scenarios,)`, > 0.
        rate: risk-free rate of each scenario, a scalar or shape `(nb_scenarios,)`.
        rebalance_every: days between hedge dates, an integer >= 1 (a numpy integer works too). 1 is daily.
        settlement: `"cash"` or `"physical"`, see `settlement.py`.
        proportional, fixed, kappa: the cost settings of `transaction_cost`, charged on every trade and on the
            settlement's closing trade.

    Returns:
        The `HedgeOutcome` of the batch.

    Raises:
        ValueError: if a shape does not match, `tau` is out of range, `rebalance_every` is not an integer >= 1,
            `settlement` is unknown, the policy returns the wrong shape, or a cost setting, `strike` or a
            settlement price is invalid (see `transaction_cost` and `settlement.py`).
    """
    if settlement not in SETTLEMENTS:
        raise ValueError(f"settlement must be one of {tuple(SETTLEMENTS)}, got {settlement!r}")
    # bool is an Integral too, but True as a number of days is a mistake
    if isinstance(rebalance_every, bool) or not isinstance(rebalance_every, numbers.Integral) or rebalance_every < 1:
        raise ValueError(f"rebalance_every must be an integer >= 1, got {rebalance_every!r}")
    rebalance_every = int(rebalance_every)
    spot = _as_float(spot)
    dtype, device = spot.dtype, spot.device
    if spot.ndim != 2 or spot.shape[0] == 0:
        raise ValueError(f"spot must have shape (nb_scenarios, n_days) with nb_scenarios >= 1, got {tuple(spot.shape)}")
    nb_scenarios, n_days = spot.shape
    inputs = _as_float(inputs, dtype)
    if inputs.ndim != 3 or inputs.shape[0] != nb_scenarios or inputs.shape[2] != n_days:
        raise ValueError(
            f"inputs must have shape ({nb_scenarios}, nb_features, {n_days}) to match spot, got {tuple(inputs.shape)}"
        )
    tau = torch.as_tensor(tau, device=device)
    if tau.is_floating_point() or tau.shape != (nb_scenarios,):
        raise ValueError(f"tau must be an int tensor of shape ({nb_scenarios},), got {tau.dtype} {tuple(tau.shape)}")
    if not torch.all((tau >= 0) & (tau < n_days)):
        raise ValueError(f"tau must be in [0, {n_days - 1}], the days of spot")
    tau = tau.long()
    per_scenario = {}
    for name, value in (("premium", premium), ("rate", rate), ("strike", strike)):
        value = torch.as_tensor(value, dtype=dtype, device=device)
        if value.shape not in ((), (nb_scenarios,)):
            raise ValueError(f"{name} must be a scalar or of shape ({nb_scenarios},), got {tuple(value.shape)}")
        per_scenario[name] = value.expand(nb_scenarios)  # a scalar becomes one value per scenario
    premium, rate, strike = per_scenario["premium"], per_scenario["rate"], per_scenario["strike"]

    days = torch.arange(n_days, device=device)
    discount = torch.exp(-rate[:, None] * days / DAYS_PER_YEAR)  # e^{-r t_d}, (nb_scenarios, n_days)
    # Every day past τ reads day τ, so a scenario is frozen at its τ: an interval past τ earns nothing and the
    # NaNs after expiry are never read. The discount is frozen too, or a held position would earn its drift.
    until_tau = torch.minimum(days, tau[:, None])
    frozen = torch.gather(spot, 1, until_tau)
    spot_disc = torch.gather(discount, 1, until_tau) * frozen
    rows = torch.arange(nb_scenarios, device=device)  # with `tau`, picks each scenario's own day τ

    position = torch.zeros(nb_scenarios, dtype=dtype, device=device)  # δ_{-1} = 0: no shares before day 0
    gains = torch.zeros(nb_scenarios, dtype=dtype, device=device)
    costs = torch.zeros(nb_scenarios, dtype=dtype, device=device)
    positions = []
    for d in range(0, int(tau.max()), rebalance_every):
        alive = d < tau
        features = torch.where(alive[:, None], inputs[:, :, d], torch.zeros((), dtype=dtype, device=device))
        proposed = policy(features, position)
        if not isinstance(proposed, torch.Tensor) or proposed.shape != (nb_scenarios,):
            got = tuple(proposed.shape) if isinstance(proposed, torch.Tensor) else type(proposed).__name__
            raise ValueError(f"policy must return a tensor of shape ({nb_scenarios},), got {got}")
        new = torch.where(alive, proposed.to(dtype), position)  # an ended scenario keeps its position
        trade_cost = transaction_cost(new - position, frozen[:, d], proportional, fixed, kappa)
        costs = costs + discount[:, d] * trade_cost
        # The position chosen on day d earns from d to the next hedge date, or to τ if that comes first.
        nxt = min(d + rebalance_every, n_days - 1)
        gains = gains + new * (spot_disc[:, nxt] - spot_disc[:, d])
        position = new
        positions.append(new)

    # Settlement at τ: the payoff owed to the holder plus the cost of closing the share position. The shares'
    # value is already in `gains`, so closing them changes the P&L only by that trade's cost.
    settle = SETTLEMENTS[settlement]
    spot_tau = frozen[rows, tau]
    settled = settle(spot_tau, position, strike, proportional, fixed, kappa)
    payoff = settle(spot_tau, position, strike)  # the same settlement without costs, to split the costs out
    discount_tau = discount[rows, tau]
    pnl = premium + gains - costs - discount_tau * settled
    costs = costs + discount_tau * (settled - payoff)  # add the closing trade's cost to the reported costs
    stacked = torch.stack(positions, dim=1) if positions else torch.zeros((nb_scenarios, 0), dtype=dtype, device=device)
    return HedgeOutcome(pnl=pnl, costs=costs, positions=stacked)
