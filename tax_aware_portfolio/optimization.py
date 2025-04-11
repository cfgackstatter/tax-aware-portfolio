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
    tax_aversion: float,
    current_prices: Dict[str, float],
    short_term_rate: float = 0.35,
    long_term_rate: float = 0.15,
    current_date: Optional[datetime] = None,
    sell_threshold: float = 0.001,
    buy_threshold: float = 0.001,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Optimizes a portfolio using mean-variance optimization with tax liability considerations.

    Args:
        portfolio (Portfolio): Portfolio object containing tax lots.
        alphas (np.ndarray): Expected returns for each asset (array matching the order of tickers).
        cov_matrix (np.ndarray): Covariance matrix for the assets.
        risk_aversion (float): Risk aversion parameter (higher values penalize risk more).
        tax_aversion (float): Tax aversion parameter (higher values penalize tax liability more).
        current_prices (Dict[str, float]): Current prices of the assets.
        short_term_rate (float): Tax rate for short-term gains (default: 0.35).
        long_term_rate (float): Tax rate for long-term gains (default: 0.15).
        current_date (Optional[datetime]): Current date for tax calculations (default: now).
        sell_threshold (float): Minimum fraction to sell for a lot to be included in results (default: 0.001).
        buy_threshold (float): Minimum fraction to buy for a ticker to be included in results (default: 0.001).
        verbose (bool): If True, prints debug information (default: False).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'weights': Optimized weights for each ticker.
            - 'current_weights': Current weights for each ticker.
            - 'objective_value': Value of the optimization objective.
            - 'status': Solver status.
            - 'sell_decisions': Details of sell decisions for each ticker.
            - 'buy_decisions': Details of buy decisions for each ticker.
            - 'tax_details': Tax-related details (gains, losses, net values, tax liability).
    """
    if current_date is None:
        current_date = datetime.now()

    # Extract tickers and create mapping
    tickers = list(portfolio.tax_lots.keys())
    n_tickers = len(tickers)

    # Calculate current portfolio value
    portfolio_value = sum(
        sum(lot.shares * current_prices[ticker] for lot in lots)
        for ticker, lots in portfolio.tax_lots.items()
    )
    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be greater than zero.")

    # Calculate current weights
    current_weights = np.zeros(n_tickers)
    for i, ticker in enumerate(tickers):
        ticker_value = sum(lot.shares * current_prices[ticker] for lot in portfolio.tax_lots[ticker])
        current_weights[i] = ticker_value / portfolio_value if portfolio_value > 0 else 0

    # Collect all tax lots and map them to tickers
    all_lots = []
    lot_to_ticker = {}
    for ticker, lots in portfolio.tax_lots.items():
        for lot in lots:
            all_lots.append(lot)
            lot_to_ticker[len(all_lots) - 1] = ticker
    n_lots = len(all_lots)

    # Create optimization variables
    sell_fractions = cp.Variable(n_lots, nonneg=True)  # Fraction of each lot to sell
    final_weights = cp.Variable(n_tickers, nonneg=True)  # Final weights for each ticker
    buy_fractions = cp.Variable(n_tickers, nonneg=True)  # Fraction of portfolio value to buy for each ticker

    # Constraints
    constraints = [
        sell_fractions <= 1,  # Can't sell more than 100% of any lot
        cp.sum(final_weights) == 1  # Fully invested portfolio
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

    # Calculate tax impacts
    st_gains, st_losses, lt_gains, lt_losses = [], [], [], []
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
                lt_losses.append(-realized_amount)
        else:
            if gain_per_share > 0:
                st_gains.append(realized_amount)
            else:
                st_losses.append(-realized_amount)

    # Sum up gains and losses
    st_total_gains = cp.sum(st_gains) if st_gains else 0
    st_total_losses = cp.sum(st_losses) if st_losses else 0
    lt_total_gains = cp.sum(lt_gains) if lt_gains else 0
    lt_total_losses = cp.sum(lt_losses) if lt_losses else 0

    # Calculate net values
    st_net = st_total_gains - st_total_losses
    lt_net = lt_total_gains - lt_total_losses

    # Create binary variables
    st_is_positive = cp.Variable(1, boolean=True)
    lt_is_positive = cp.Variable(1, boolean=True)

    # Create variables for net gains/losses
    st_net_gain = cp.Variable(1)
    st_net_loss = cp.Variable(1)
    lt_net_gain = cp.Variable(1)
    lt_net_loss = cp.Variable(1)

    # Big-M constant
    M = portfolio_value  # Large number but not too large to avoid numerical issues

    # Short-term net constraints
    constraints.append(st_net <= M * st_is_positive)  # If st_net > 0, st_is_positive must be 1
    constraints.append(st_net >= -M * (1 - st_is_positive))  # If st_net < 0, st_is_positive must be 0

    # For st_net_gain
    constraints.append(st_net_gain <= st_net + M * (1 - st_is_positive))  # Upper bound
    constraints.append(st_net_gain >= st_net - M * (1 - st_is_positive))  # Lower bound
    constraints.append(st_net_gain <= M * st_is_positive)  # Force to 0 if st_net < 0

    # For st_net_loss
    constraints.append(st_net_loss <= -st_net + M * st_is_positive)  # Upper bound
    constraints.append(st_net_loss >= -st_net - M * st_is_positive)  # Lower bound
    constraints.append(st_net_loss <= M * (1 - st_is_positive))  # Force to 0 if st_net > 0

    # Long-term net constraints (same logic as short-term)
    constraints.append(lt_net <= M * lt_is_positive)
    constraints.append(lt_net >= -M * (1 - lt_is_positive))

    constraints.append(lt_net_gain <= lt_net + M * (1 - lt_is_positive))
    constraints.append(lt_net_gain >= lt_net - M * (1 - lt_is_positive))
    constraints.append(lt_net_gain <= M * lt_is_positive)

    constraints.append(lt_net_loss <= -lt_net + M * lt_is_positive)
    constraints.append(lt_net_loss >= -lt_net - M * lt_is_positive)
    constraints.append(lt_net_loss <= M * (1 - lt_is_positive))
    
    # Variables for final taxable amounts - remove nonneg=True to allow negative values
    st_taxable = cp.Variable(1)  # Allow negative values
    lt_taxable = cp.Variable(1)  # Allow negative values

    # Binary variables for when net values after netting are positive or negative
    st_result_positive = cp.Variable(1, boolean=True)
    lt_result_positive = cp.Variable(1, boolean=True)

    # For short-term result (st_net_gain - lt_net_loss)
    constraints.append(st_net_gain - lt_net_loss <= M * st_result_positive)
    constraints.append(st_net_gain - lt_net_loss >= -M * (1 - st_result_positive))

    # For long-term result (lt_net_gain - st_net_loss)
    constraints.append(lt_net_gain - st_net_loss <= M * lt_result_positive)
    constraints.append(lt_net_gain - st_net_loss >= -M * (1 - lt_result_positive))

    # Set st_taxable to the exact value of st_net_gain - lt_net_loss or negative st_net_loss
    constraints.append(st_taxable <= (st_net_gain - lt_net_loss) + M * (1 - st_result_positive))
    constraints.append(st_taxable >= (st_net_gain - lt_net_loss) - M * (1 - st_result_positive))

    constraints.append(st_taxable <= -st_net_loss + M * st_result_positive)
    constraints.append(st_taxable >= -st_net_loss - M * st_result_positive)

    # Set lt_taxable to the exact value of lt_net_gain - st_net_loss or negative lt_net_loss
    constraints.append(lt_taxable <= (lt_net_gain - st_net_loss) + M * (1 - lt_result_positive))
    constraints.append(lt_taxable >= (lt_net_gain - st_net_loss) - M * (1 - lt_result_positive))

    constraints.append(lt_taxable <= -lt_net_loss + M * lt_result_positive)
    constraints.append(lt_taxable >= -lt_net_loss - M * lt_result_positive)

    # Calculate final tax liability
    tax_liability = cp.Variable(1)  # Allow positive or negative values
    tax_liability = short_term_rate * st_taxable + long_term_rate * lt_taxable
    #constraints.append(tax_liability == short_term_rate * st_taxable + long_term_rate * lt_taxable)
    
    # Mean-variance objective with tax penalty
    portfolio_return = final_weights @ alphas
    portfolio_risk = cp.quad_form(final_weights, cov_matrix)
    objective = cp.Maximize(portfolio_return - tax_aversion * tax_liability / portfolio_value - risk_aversion * portfolio_risk)
    
    # Solve the problem with SCIP solver
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='SCIP', verbose=verbose)
    
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
        
        if sell_fractions.value[i] > sell_threshold:  # Only include meaningful sells
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
        if buy_fractions.value[i] > buy_threshold:  # Only include meaningful buys
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