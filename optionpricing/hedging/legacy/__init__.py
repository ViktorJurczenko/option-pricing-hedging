"""
The legacy hedging environment, kept as a library for the SAC agent:
HedgingEnv (env), its value functions for the accounting reward
(option_value), the analytic baselines (baselines) and the P&L evaluation
(evaluation). Its exercise dates and deltas come from Longstaff-Schwartz, and
its P&L has two open accounting bugs (.claude/BUGS.md, B-037 and B-038). Only `train-hedge-legacy` and
`test-hedge-legacy` use it. Kept to reproduce the earlier SAC results.
"""
