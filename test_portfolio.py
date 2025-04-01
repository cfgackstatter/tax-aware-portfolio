from tax_aware_portfolio.tax_lot import TaxLot
from tax_aware_portfolio.portfolio import Portfolio
from datetime import datetime

def test_portfolio():
    portfolio = Portfolio()

    # Buy shares
    portfolio.buy("AAPL", 100, 150.0, datetime(2023, 1, 1))
    portfolio.buy("AAPL", 50, 160.0, datetime(2023, 6, 1))

    # Sell shares using FIFO method
    sold_fifo = portfolio.sell("AAPL", 120, method='FIFO')
    print("Sold (FIFO):", sold_fifo)

    # Sell shares using HIFO method
    sold_hifo = portfolio.sell("AAPL", 30, method='HIFO')
    print("Sold (HIFO):", sold_hifo)

    # Calculate tax burden (example logic)
    current_price = 180.0
    tax_burden_fifo = portfolio.calculate_tax_burden(sold_fifo, current_price)
    print("Tax Burden (FIFO):", tax_burden_fifo)

    tax_burden_hifo = portfolio.calculate_tax_burden(sold_hifo, current_price)
    print("Tax Burden (HIFO):", tax_burden_hifo)

if __name__ == "__main__":
    test_portfolio()