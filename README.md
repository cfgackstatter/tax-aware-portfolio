# Tax-Aware Portfolio

This Python package provides tools for managing a stock portfolio with tax considerations. It includes functionality for tracking tax lots, buying and selling shares using different methods (FIFO, HIFO), and calculating tax burdens.

## Features

- Track individual tax lots with purchase date, price, and number of shares
- Buy and sell shares using FIFO (First-In-First-Out) or HIFO (Highest-In-First-Out) methods
- Calculate tax burdens based on selling strategies

## Usage

```
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

## Installation

Clone the repository and install the package:
```
git clone https://github.com/cfgackstatter/tax-aware-portfolio.git
cd tax-aware-portfolio
pip install -e .
```

## Testing

Run the test file:

```python test_portfolio.py```

## License

This project is licensed under the MIT License.