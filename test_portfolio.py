from tax_aware_portfolio.portfolio import TaxLot, Portfolio
from datetime import datetime, timedelta

def test_portfolio():
    portfolio = Portfolio()
    current_date = datetime(2025, 4, 1)  # Current date from the system

    # Buy shares of multiple stocks
    portfolio.buy("AAPL", 100, 150.0, current_date - timedelta(days=400))  # Long-term
    portfolio.buy("AAPL", 50, 160.0, current_date - timedelta(days=200))   # Short-term
    portfolio.buy("MSFT", 75, 250.0, current_date - timedelta(days=500))   # Long-term
    portfolio.buy("MSFT", 25, 280.0, current_date - timedelta(days=100))   # Short-term
    portfolio.buy("GOOG", 10, 2500.0, current_date - timedelta(days=50))   # Short-term with loss potential
    
    # Current prices for calculations
    current_prices = {
        "AAPL": 180.0,
        "MSFT": 300.0,
        "GOOG": 2400.0  # Loss for tax-loss harvesting
    }
    
    print("Portfolio after buying:")
    for ticker, lots in portfolio.tax_lots.items():
        print(f"{ticker}: {sum(lot.shares for lot in lots)} shares")
    
    # Test regular FIFO selling
    print("\n--- Testing FIFO selling ---")
    sold_fifo = portfolio.sell("AAPL", 120, current_prices["AAPL"], strategy='FIFO', tax_date=current_date)
    print(f"Sold {sum(lot.shares for lot in sold_fifo)} shares of AAPL using FIFO")
    
    # Test HIFO selling
    print("\n--- Testing HIFO selling ---")
    sold_hifo = portfolio.sell("MSFT", 30, current_prices["MSFT"], strategy='HIFO', tax_date=current_date)
    print(f"Sold {sum(lot.shares for lot in sold_hifo)} shares of MSFT using HIFO")
    
    # Test tax-efficient selling
    print("\n--- Testing tax-efficient selling ---")
    sold_tax_efficient = portfolio.sell("MSFT", 40, current_prices["MSFT"], strategy='TAX_EFFICIENT', tax_date=current_date)
    print(f"Sold {sum(lot.shares for lot in sold_tax_efficient)} shares of MSFT using tax-efficient method")
    
    # Test tax loss harvesting opportunities
    print("\n--- Testing tax loss harvesting opportunities ---")
    opportunities = portfolio.find_tax_loss_harvesting_opportunities(current_prices, threshold=-500)
    if opportunities:
        print(f"Found {len(opportunities)} tax loss harvesting opportunities:")
        for opp in opportunities:
            print(f"  {opp['ticker']}: {opp['shares']} shares, unrealized loss: ${opp['unrealized_loss']:.2f}")
    else:
        print("No tax loss harvesting opportunities found")
    
    # Test portfolio summary
    print("\n--- Portfolio Summary ---")
    summary = portfolio.get_portfolio_summary(current_prices)
    print(f"Total Value: ${summary['total_value']:.2f}")
    print(f"Cost Basis: ${summary['cost_basis']:.2f}")
    print(f"Unrealized Gains: ${summary['unrealized_gains']:.2f}")
    print(f"Realized Short-Term Gains: ${summary['realized_gains']['short_term']:.2f}")
    print(f"Realized Long-Term Gains: ${summary['realized_gains']['long_term']:.2f}")
    
    # Test tax burden calculation
    print("\n--- Tax Burden Calculation ---")
    tax_burden = portfolio.calculate_tax_burden(sold_fifo, current_prices["AAPL"], tax_date=current_date)
    print(f"Short-term gains: ${tax_burden['short_term_gains']:.2f}")
    print(f"Long-term gains: ${tax_burden['long_term_gains']:.2f}")
    print(f"Total tax burden: ${tax_burden['total_tax_burden']:.2f}")
    
    # Check realized gains tracking
    print("\n--- Realized Gains ---")
    print(f"Realized Short-Term Gains: ${portfolio.realized_short_term_gains:.2f}")
    print(f"Realized Long-Term Gains: ${portfolio.realized_long_term_gains:.2f}")
    
    # Test wash sale detection
    print("\n--- Wash Sale Detection ---")
    mock_sold_lot = TaxLot("GOOG", 5, 2500.0, current_date - timedelta(days=50))
    # Add sale_date attribute
    mock_sold_lot.sale_date = current_date
    
    is_wash_sale = portfolio.check_wash_sales(mock_sold_lot, current_date + timedelta(days=15), "GOOG")
    print(f"Is Wash Sale: {is_wash_sale}")

if __name__ == "__main__":
    test_portfolio()