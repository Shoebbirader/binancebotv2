import numpy as np
import pandas as pd

class AdvancedRiskManager:
    """
    Advanced risk management: drawdown, VaR, expected shortfall, Monte Carlo
    """
    def __init__(self, max_drawdown_pct=20):
        self.max_drawdown_pct = max_drawdown_pct
        self.equity_curve = []

    def update_equity(self, equity):
        self.equity_curve.append(equity)

    def current_drawdown(self):
        if not self.equity_curve:
            return 0
        peak = np.max(self.equity_curve)
        trough = self.equity_curve[-1]
        drawdown = (peak - trough) / peak * 100
        return drawdown

    def check_drawdown_limit(self):
        dd = self.current_drawdown()
        if dd > self.max_drawdown_pct:
            print(f"Drawdown {dd:.2f}% exceeds limit! Reduce positions.")
            return True
        return False

    def value_at_risk(self, returns, confidence=0.95):
        var = -np.percentile(returns, 100 * (1 - confidence))
        print(f"VaR at {confidence*100:.1f}%: {var:.4f}")
        return var

    def expected_shortfall(self, returns, confidence=0.95):
        var = self.value_at_risk(returns, confidence)
        es = -returns[returns < -var].mean()
        print(f"Expected Shortfall at {confidence*100:.1f}%: {es:.4f}")
        return es

    def monte_carlo_simulation(self, returns, n_sim=1000, horizon=30):
        sims = np.zeros((n_sim, horizon))
        for i in range(n_sim):
            sims[i] = np.random.choice(returns, size=horizon, replace=True)
        sim_paths = sims.cumsum(axis=1)
        worst_case = np.min(sim_paths[:, -1])
        print(f"Monte Carlo worst-case loss over {horizon} days: {worst_case:.4f}")
        return sim_paths

# Example usage:
# risk_mgr = AdvancedRiskManager(max_drawdown_pct=15)
# risk_mgr.update_equity(current_equity)
# risk_mgr.check_drawdown_limit()
# returns = np.array([...])
# risk_mgr.value_at_risk(returns)
# risk_mgr.expected_shortfall(returns)
# risk_mgr.monte_carlo_simulation(returns)
