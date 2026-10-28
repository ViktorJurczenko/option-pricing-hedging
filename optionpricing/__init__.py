"""
Option pricing and hedging research code.

- core: the GBM path simulator, the pricer base class and generic metrics.
- pricing: European (Black-Scholes, Monte-Carlo) and American (binomial
  trees, Longstaff-Schwartz) pricers.
- hedging: the hedging environment, option value functions, baselines and
  P&L evaluation, independent of any learner.
- ml: everything learned: the MLP surrogate pricer, the SAC hedging agent
  and the experiment drivers.
"""
