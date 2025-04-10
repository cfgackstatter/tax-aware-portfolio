import numpy as np
from datetime import datetime, timedelta
import cvxpy as cp

# Import the Portfolio and TaxLot classes
from tax_aware_portfolio.portfolio import Portfolio, TaxLot
from tax_aware_portfolio.optimization import optimize_mean_variance

# Create a small portfolio with tax lots
portfolio = Portfolio()
current_date = datetime(2025, 4, 9)

# Add tax lots for two tickers
portfolio.buy("AAPL", 10, 190.0, current_date - timedelta(days=400))  # Long-term
portfolio.buy("AAPL", 5, 160.0, current_date - timedelta(days=200))   # Short-term
portfolio.buy("MSFT", 8, 250.0, current_date - timedelta(days=500))   # Long-term
portfolio.buy("MSFT", 2, 280.0, current_date - timedelta(days=100))   # Short-term

# Define expected returns (alphas) and covariance matrix
alphas = np.array([0.08, 0.12])  # Expected returns for AAPL and MSFT
cov_matrix = np.array([
    [0.1, 0.03],
    [0.03, 0.15]
])  # Covariance matrix

# Current prices for the assets
current_prices = {
    "AAPL": 180.0,
    "MSFT": 300.0
}

# Risk aversion parameter
risk_aversion = 2.0

# Run the optimization
result = optimize_mean_variance(
    portfolio=portfolio,
    alphas=alphas,
    cov_matrix=cov_matrix,
    risk_aversion=risk_aversion,
    current_prices=current_prices,
    short_term_rate=0.35,
    long_term_rate=0.15,
    current_date=current_date
)

# Print the results
print("Optimized weights:")
for ticker, weight in result['weights'].items():
    print(f"{ticker}: {weight:.4f}")

print("\nSell decisions:")
for ticker, decisions in result['sell_decisions'].items():
    for decision in decisions:
        print(f"{ticker}: Sell {decision['shares_to_sell']:.2f} shares, " 
              f"realized gain/loss: ${decision['realized_gain_loss']:.2f}")

print("\nBuy decisions:")
for ticker, decision in result['buy_decisions'].items():
    print(f"{ticker}: Buy {decision['shares']:.2f} shares, amount: ${decision['amount']:.2f}")

print("\nTax details:")
tax_details = result['tax_details']
print(f"Short-term gains: ${tax_details['short_term_gains']:.2f}")
print(f"Long-term gains: ${tax_details['long_term_gains']:.2f}")
print(f"Total tax liability: ${tax_details['tax_liability'].item():.2f}")