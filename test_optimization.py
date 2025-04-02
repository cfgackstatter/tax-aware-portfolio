import cvxpy as cp
import numpy as np
from datetime import datetime, timedelta
from tax_aware_portfolio. portfolio import Portfolio

def optimize_portfolio_with_taxes(portfolio, expected_returns, risk_model, current_prices,
                                  risk_aversion=1.0, short_term_rate=0.35, long_term_rate=0.15):
    """
    Optimize portfolio considering tax implications.

    Args:
        portfolio: Your Portfolio object
        expected_returns: Expected returns for each asset
        risk_model: Covariance matrix
        current_prices: Current prices of assets
        risk_aversion: Risk aversion parameter
        short_term_rate: Short-term capital gains tax rate
        long_term_rate: Long-term capital gains tax rate

    Returns:
        Optimal weights for each asset
    """
    # Get current portfolio state
    summary = portfolio.get_portfolio_summary(current_prices)
    tickers = list(summary['positions'].keys())
    n_assets = len(tickers)

    # Current weights
    current_values = [summary['positions'][ticker]['current_value'] for ticker in tickers]
    total_value = sum(current_values)
    current_weights = np.array([val/total_value for val in current_values])

    # Extract tax lot information
    tax_lots_by_ticker = {}
    for ticker in tickers:
        tax_lots_by_ticker[ticker] = list(portfolio.tax_lots[ticker])

    # Calculate potential tax implications of selling
    potential_tax_impact = []
    for i, ticker in enumerate(tickers):
        # For each ticker, estimate tax impact per unit of weight reduction
        lots = tax_lots_by_ticker[ticker]
        if not lots:
            potential_tax_impact.append(0)
            continue

        # Sort lots by tax efficiency (FIFO, HIFO, or tax-efficient)
        # This depends on your selling strategy
        sorted_lots = sorted(lots, key=lambda x: (
            0 if (current_prices[ticker] - x.purchase_price) < 0 else 1, # Losses first
            0 if (datetime.now() - x.purchase_date).days >= 365 else 1, # Long-term gains next
            current_prices[ticker] - x.purchase_price # Smallest gains last
        ))

        # Calculate weighted average tax rate for this ticker
        total_shares = sum(lot.shares for lot in lots)
        weighted_tax_rate = 0
        for lot in sorted_lots:
            gain_per_share = current_prices[ticker] - lot.purchase_price
            is_long_term = (datetime.now() - lot.purchase_date).days >= 365
            tax_rate = long_term_rate if is_long_term else short_term_rate
            
            if gain_per_share > 0:  # Only positive gains are taxed
                weighted_tax_rate += (lot.shares / total_shares) * tax_rate * gain_per_share
                
        potential_tax_impact.append(weighted_tax_rate)
    
    # Convert to numpy array
    potential_tax_impact = np.array(potential_tax_impact)
    
    # Define optimization variables
    w = cp.Variable(n_assets)
    
    # Define objective function with tax considerations
    returns = w @ expected_returns
    risk = cp.quad_form(w, risk_model)
    tax_impact = cp.sum(cp.multiply(cp.maximum(0, current_weights - w), potential_tax_impact))
    
    # Mean-variance objective with tax penalty
    objective = returns - risk_aversion * risk - tax_impact
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Fully invested
        w >= 0           # No short selling
    ]
    
    # Solve the problem
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()
    
    # Return optimal weights
    return {ticker: w.value[i] for i, ticker in enumerate(tickers)}


# Initialize your portfolio
portfolio = Portfolio()

# Buy shares of multiple stocks
portfolio.buy("AAPL", 100, 150.0, datetime.now() - timedelta(days=400))  # Long-term
portfolio.buy("AAPL", 50, 160.0, datetime.now() - timedelta(days=200))   # Short-term
portfolio.buy("MSFT", 75, 250.0, datetime.now() - timedelta(days=500))   # Long-term
portfolio.buy("MSFT", 25, 280.0, datetime.now() - timedelta(days=100))   # Short-term
portfolio.buy("GOOG", 10, 2500.0, datetime.now() - timedelta(days=50))   # Short-term with loss potential

# Define expected returns and risk model
expected_returns = np.array([0.08, 0.12, 0.10])  # Example expected returns
risk_model = np.array([  # Example covariance matrix
    [0.1, 0.03, 0.05],
    [0.03, 0.15, 0.06],
    [0.05, 0.06, 0.12]
])

# Current prices
current_prices = {
    "AAPL": 180.0,
    "MSFT": 300.0,
    "GOOG": 2400.0
}

# Run the optimization
optimal_weights = optimize_portfolio_with_taxes(
    portfolio, 
    expected_returns, 
    risk_model, 
    current_prices,
    risk_aversion=2.0
)

print("Optimal portfolio weights:", optimal_weights)