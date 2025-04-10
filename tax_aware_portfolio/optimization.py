from typing import Dict, List, Optional, Any
import numpy as np
import cvxpy as cp
from datetime import datetime
from tax_aware_portfolio.portfolio import Portfolio


def optimize_mean_variance(
    portfolio: Portfolio,
    alphas: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float,
    current_prices: Dict[str, float],
    short_term_rate: float = 0.35,
    long_term_rate: float = 0.15,
    current_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
    """
    Optimizes a portfolio using mean-variance optimization with tax liability considerations.
    
    Args:
        portfolio: Portfolio object containing tax lots
        alphas: Expected returns for each asset (array matching the order of tickers)
        cov_matrix: Covariance matrix for the assets
        risk_aversion: Risk aversion parameter
        current_prices: Current prices of the assets
        short_term_rate: Tax rate for short-term gains
        long_term_rate: Tax rate for long-term gains
        current_date: Current date for tax calculations
        
    Returns:
        A dictionary containing the optimized weights, buy/sell decisions, and tax liability.
    """
    if current_date is None:
        current_date = datetime.now()

    # Extract tickers and create mapping
    tickers = list(portfolio.tax_lots.keys())
    ticker_indices = {ticker: i for i, ticker in enumerate(tickers)}
    n_tickers = len(tickers)

    # Calculate current portfolio value
    portfolio_value = sum(
        sum(lot.shares * current_prices[ticker] for lot in lots)
        for ticker, lots in portfolio.tax_lots.items()
    )

    # Calculate current weights
    current_weights = np.zeros(n_tickers)
    for i, ticker in enumerate(tickers):
        ticker_value = sum(lot.shares * current_prices[ticker] for lot in portfolio.tax_lots[ticker])
        current_weights[i] = ticker_value / portfolio_value if portfolio_value > 0 else 0

    # Collect all tax lots
    all_lots = []
    lot_to_ticker = {}
    for ticker, lots in portfolio.tax_lots.items():
        for lot in lots:
            all_lots.append(lot)
            lot_to_ticker[len(all_lots)-1] = ticker
    n_lots = len(all_lots)

    # Create variables for sell fractions (0 to 1 for each lot)
    sell_fractions = cp.Variable(n_lots, nonneg=True)

    # Create variables for final weights
    final_weights = cp.Variable(n_tickers, nonneg=True)

    # Create variables for buy amounts (as fraction of portfolio value)
    buy_fractions = cp.Variable(n_tickers, nonneg=True)

    # Constraints
    constraints = [
        sell_fractions <= 1,  # Can't sell more than 100% of any lot
        cp.sum(final_weights) == 1  # Fully invested
    ]

    # Link sell fractions to final weights
    for i, ticker in enumerate(tickers):
        ticker_value = sum(lot.shares * current_prices[ticker] for lot in portfolio.tax_lots[ticker])
        ticker_lots = [j for j in range(n_lots) if lot_to_ticker[j] == ticker]
        
        # Calculate value sold from this ticker
        sold_value = cp.sum([
            sell_fractions[j] * all_lots[j].shares * current_prices[ticker]
            for j in ticker_lots
        ]) if ticker_lots else 0

        # Final weight constraint: (current value - sold value + buy value) / portfolio value
        constraints.append(
            final_weights[i] == (ticker_value - sold_value + buy_fractions[i] * portfolio_value) / portfolio_value
        )

    # Cash constraint: total buys must equal total sells
    total_sells = cp.sum([
        sell_fractions[j] * all_lots[j].shares * current_prices[lot_to_ticker[j]]
        for j in range(n_lots)
    ])
    total_buys = cp.sum(buy_fractions) * portfolio_value
    constraints.append(total_buys == total_sells)

    # Calculate tax impacts with proper netting using binary variables
    # First, separate gains and losses by type
    st_gains = []
    st_losses = []
    lt_gains = []
    lt_losses = []

    for i, lot in enumerate(all_lots):
        ticker = lot_to_ticker[i]
        gain_per_share = current_prices[ticker] - lot.purchase_price
        is_long_term = (current_date - lot.purchase_date).days >= 365
        
        # Calculate realized gain/loss for this lot
        realized_amount = gain_per_share * lot.shares * sell_fractions[i]
        
        if is_long_term:
            if gain_per_share > 0:
                lt_gains.append(realized_amount)
            else:
                lt_losses.append(-realized_amount)  # Make positive for easier calculation
        else:
            if gain_per_share > 0:
                st_gains.append(realized_amount)
            else:
                st_losses.append(-realized_amount)  # Make positive for easier calculation

    # Sum up gains and losses
    st_total_gains = cp.sum(st_gains) if st_gains else 0
    st_total_losses = cp.sum(st_losses) if st_losses else 0
    lt_total_gains = cp.sum(lt_gains) if lt_gains else 0
    lt_total_losses = cp.sum(lt_losses) if lt_losses else 0

    # Create binary variables for tax netting logic
    st_net_positive = cp.Variable(1, boolean=True)
    lt_net_positive = cp.Variable(1, boolean=True)

    # Calculate net values
    st_net = st_total_gains - st_total_losses
    lt_net = lt_total_gains - lt_total_losses

    # Big-M constant for logical constraints
    M = 1e6  # A large constant

    # Short-term net gain/loss logic
    constraints.append(st_net <= M * st_net_positive)
    constraints.append(st_net >= -M * (1 - st_net_positive))

    # Long-term net gain/loss logic
    constraints.append(lt_net <= M * lt_net_positive)
    constraints.append(lt_net >= -M * (1 - lt_net_positive))

    # Variables for net gains and losses after first netting
    st_net_gain = cp.Variable(1, nonneg=True)
    st_net_loss = cp.Variable(1, nonneg=True)
    lt_net_gain = cp.Variable(1, nonneg=True)
    lt_net_loss = cp.Variable(1, nonneg=True)

    # Constraints for net gains and losses
    constraints.append(st_net_gain <= M * st_net_positive)
    constraints.append(st_net_loss <= M * (1 - st_net_positive))
    constraints.append(lt_net_gain <= M * lt_net_positive)
    constraints.append(lt_net_loss <= M * (1 - lt_net_positive))

    constraints.append(st_net_gain <= st_net + M * (1 - st_net_positive))
    constraints.append(st_net_loss <= -st_net + M * st_net_positive)
    constraints.append(lt_net_gain <= lt_net + M * (1 - lt_net_positive))
    constraints.append(lt_net_loss <= -lt_net + M * lt_net_positive)

    # Binary variables for remaining gains after cross-netting
    st_has_remaining = cp.Variable(1, boolean=True)
    lt_has_remaining = cp.Variable(1, boolean=True)
    
    # Constraints for remaining gains
    constraints.append(st_net_gain - lt_net_loss <= M * st_has_remaining)
    constraints.append(lt_net_gain - st_net_loss <= M * lt_has_remaining)
    
    # Variables for final taxable amounts
    st_taxable = cp.Variable(1, nonneg=True)
    lt_taxable = cp.Variable(1, nonneg=True)

    # Constraints for final taxable amounts
    constraints.append(st_taxable <= M * st_has_remaining)
    constraints.append(lt_taxable <= M * lt_has_remaining)
    
    constraints.append(st_taxable <= st_net_gain - lt_net_loss + M * (1 - st_has_remaining))
    constraints.append(lt_taxable <= lt_net_gain - st_net_loss + M * (1 - lt_has_remaining))

    # Calculate final tax liability
    tax_liability = short_term_rate * st_taxable + long_term_rate * lt_taxable
    
    # Mean-variance objective with tax penalty
    portfolio_return = final_weights @ alphas
    portfolio_risk = cp.quad_form(final_weights, cov_matrix)

    # Objective: maximize return - tax liability - risk penalty
    objective = cp.Maximize(portfolio_return - tax_liability - risk_aversion * portfolio_risk)
    
    # Solve the problem with SCIP solver
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='SCIP')
    
    # Extract results
    result = {
        'weights': {ticker: final_weights.value[i] for i, ticker in enumerate(tickers)},
        'current_weights': {ticker: current_weights[i] for i, ticker in enumerate(tickers)},
        'objective_value': problem.value,
        'status': problem.status
    }
    
    # Extract sell decisions
    sell_decisions = {}
    for i, lot in enumerate(all_lots):
        ticker = lot_to_ticker[i]
        if ticker not in sell_decisions:
            sell_decisions[ticker] = []
        
        if sell_fractions.value[i] > 0.001:  # Only include meaningful sells
            sell_decisions[ticker].append({
                'lot': lot,
                'fraction_to_sell': sell_fractions.value[i],
                'shares_to_sell': lot.shares * sell_fractions.value[i],
                'realized_gain_loss': (current_prices[ticker] - lot.purchase_price) *
                                     lot.shares * sell_fractions.value[i]
            })
    
    # Extract buy decisions
    buy_decisions = {}
    for i, ticker in enumerate(tickers):
        if buy_fractions.value[i] > 0.001:  # Only include meaningful buys
            buy_amount = buy_fractions.value[i] * portfolio_value
            buy_decisions[ticker] = {
                'amount': buy_amount,
                'shares': buy_amount / current_prices[ticker]
            }
    
    result['sell_decisions'] = sell_decisions
    result['buy_decisions'] = buy_decisions
    
    # Calculate tax details
    result['tax_details'] = {
        'short_term_gains': st_total_gains.value if hasattr(st_total_gains, 'value') else st_total_gains,
        'short_term_losses': st_total_losses.value if hasattr(st_total_losses, 'value') else st_total_losses,
        'long_term_gains': lt_total_gains.value if hasattr(lt_total_gains, 'value') else lt_total_gains,
        'long_term_losses': lt_total_losses.value if hasattr(lt_total_losses, 'value') else lt_total_losses,
        'net_short_term': st_net.value if hasattr(st_net, 'value') else st_net,
        'net_long_term': lt_net.value if hasattr(lt_net, 'value') else lt_net,
        'tax_liability': tax_liability.value if hasattr(tax_liability, 'value') else tax_liability
    }
    
    return result