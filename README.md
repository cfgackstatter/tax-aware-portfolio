# Tax-Aware Portfolio (Alpha Version)
This Python package provides tools for managing a stock portfolio with tax considerations. It includes functionality for tracking tax lots, buying and selling shares using different methods (FIFO, HIFO), and calculating tax burdens.

**Note: This is an alpha version. The API and functionality may change significantly in future releases.**

## Features
- Track individual tax lots with purchase date, price, and number of shares
- Buy and sell shares using FIFO (First-In-First-Out) or HIFO (Highest-In-First-Out) methods
- Calculate tax burdens based on selling strategies
- Tax-aware portfolio optimization using mean-variance framework

## Usage
```python
from tax_aware_portfolio.portfolio import Portfolio
from datetime import datetime

portfolio = Portfolio()
```

### Buy shares
```portfolio.buy("AAPL", 100, 150.0, datetime(2023, 1, 1))```

### Sell shares
```sold_lots = portfolio.sell("AAPL", 50, method='FIFO')```

### Calculate tax burden
tax_burden = portfolio.calculate_tax_burden(sold_lots, current_price=180.0)

### Portfolio Optimization
```python
from tax_aware_portfolio.optimization import optimize_mean_variance

result = optimize_mean_variance(
    portfolio=portfolio,
    alphas=expected_returns,
    cov_matrix=covariance_matrix,
    risk_aversion=2.0,
    current_prices=current_prices
)
```

## Installation
Clone the repository and install the package:
```
git clone https://github.com/cfgackstatter/tax-aware-portfolio.git
cd tax-aware-portfolio
pip install -e .
```

## Testing
Run the test files:
```python test_portfolio.py```
```python test_optimization.py```