"""
The legacy hedging environment, kept as a library for the SAC agent:
HedgingEnv (env), its value functions for the accounting reward
(option_value), the analytic baselines (baselines) and the P&L evaluation
(evaluation). Its exercise dates and deltas come from Longstaff-Schwartz, and
its P&L has two open accounting bugs (.claude/BUGS.md, "HedgingEnv applies
each hedge one step late" and "Hedging P&L has no cash account, so the delta
baseline hedges the wrong liability"). Only `train-hedge-legacy` and
`test-hedge-legacy` use it. Kept to reproduce the earlier SAC results.
"""
