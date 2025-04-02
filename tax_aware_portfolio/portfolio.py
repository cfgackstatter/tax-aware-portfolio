from collections import deque, defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass

@dataclass
class TaxLot:
    identifier: str
    shares: float
    purchase_price: float
    purchase_date: datetime
    sale_date: Optional[datetime] = None

class Portfolio:
    def __init__(self):
        self.tax_lots = defaultdict(deque)
        self.realized_short_term_gains = 0
        self.realized_long_term_gains = 0
        self.tax_events = []  # History of all tax events

    def __repr__(self):
        return f"Portfolio({self.tax_lots})"
    
    def buy(self, identifier: str, shares: float, purchase_price: float, purchase_date: datetime) -> None:
        """Add new shares to the portfolio."""
        self.tax_lots[identifier].append(TaxLot(identifier, shares, purchase_price, purchase_date))

    def _calculate_gain(self, lot: TaxLot, current_price: float) -> float:
        """Helper method to calculate gain/loss for a lot."""
        return (current_price - lot.purchase_price) * lot.shares
    
    def _is_long_term(self, lot: TaxLot, tax_date: datetime) -> bool:
        """Helper method to determine if a lot qualifies for long-term capital gains."""
        holding_period = (tax_date - lot.purchase_date).days
        return holding_period >= 365
    
    def sell(self, identifier: str, shares_to_sell: float, current_price: float, 
             strategy: str = 'FIFO', tax_date: Optional[datetime] = None) -> List[TaxLot]:
        """
        Sell shares using the specified strategy.
        
        Args:
            identifier: Stock ticker/identifier
            shares_to_sell: Number of shares to sell
            current_price: Current price per share
            strategy: Selling strategy ('FIFO', 'HIFO', or 'TAX_EFFICIENT')
            tax_date: Date to use for tax calculations (defaults to current date)
        
        Returns:
            List of sold tax lots
        """
        if tax_date is None:
            tax_date = datetime.now()

        if identifier not in self.tax_lots or sum(lot.shares for lot in self.tax_lots[identifier]) < shares_to_sell:
            raise ValueError("Insufficient shares to sell")
        
        sold_lots = []
        remaining_shares = shares_to_sell

        if strategy == 'FIFO':
            # First-In-First-Out
            while remaining_shares > 0:
                lot = self.tax_lots[identifier][0]
                if lot.shares <= remaining_shares:
                    sold_lots.append(self.tax_lots[identifier].popleft())
                    remaining_shares -= lot.shares
                else:
                    lot.shares -= remaining_shares
                    sold_lots.append(TaxLot(lot.identifier, remaining_shares, lot.purchase_price, lot.purchase_date))
                    remaining_shares = 0

        elif strategy == 'HIFO':
            # Highest-In-First-Out
            sorted_lots = sorted(self.tax_lots[identifier], key=lambda x: x.purchase_price, reverse=True)
            for lot in sorted_lots:
                if remaining_shares <= 0:
                    break
                if lot.shares <= remaining_shares:
                    sold_lots.append(lot)
                    self.tax_lots[identifier].remove(lot)
                    remaining_shares -= lot.shares
                else:
                    lot.shares -= remaining_shares
                    sold_lots.append(TaxLot(lot.identifier, remaining_shares, lot.purchase_price, lot.purchase_date))
                    remaining_shares = 0

        elif strategy == 'TAX_EFFICIENT':
            # Tax-efficient selling
            lot_metrics = []
            for lot in self.tax_lots[identifier]:
                gain = self._calculate_gain(lot, current_price)
                is_long_term = self._is_long_term(lot, tax_date)
                lot_metrics.append({
                    "lot": lot,
                    "gain": gain,
                    "is_long_term": is_long_term,
                    "gain_per_share": current_price - lot.purchase_price
                })

            # Sort by tax efficiency: losses first, then long-term gains, then smallest gains
            lot_metrics.sort(key=lambda x: (
                0 if x["gain"] < 0 else 1,  # Losses first
                0 if x["is_long_term"] else 1,  # Then long-term gains
                x["gain_per_share"]  # Then smallest gains
            ))
            
            for metric in lot_metrics:
                lot = metric["lot"]
                if remaining_shares <= 0:
                    break
                if lot.shares <= remaining_shares:
                    sold_lots.append(lot)
                    self.tax_lots[identifier].remove(lot)
                    remaining_shares -= lot.shares
                else:
                    new_lot = TaxLot(lot.identifier, remaining_shares, lot.purchase_price, lot.purchase_date)
                    lot.shares -= remaining_shares
                    sold_lots.append(new_lot)
                    remaining_shares = 0

        # Record tax events for all selling strategies
        self.record_tax_event(sold_lots, current_price, tax_date)
        return sold_lots
    
    def record_tax_event(self, sold_lots: List[TaxLot], current_price: float, 
                         tax_date: Optional[datetime] = None) -> None:
        """Record tax events for sold lots."""
        if tax_date is None:
            tax_date = datetime.now()
            
        for lot in sold_lots:
            gain = self._calculate_gain(lot, current_price)
            is_long_term = self._is_long_term(lot, tax_date)
            holding_period = (tax_date - lot.purchase_date).days
            
            if is_long_term:
                self.realized_long_term_gains += gain
            else:
                self.realized_short_term_gains += gain
                
            # Set the sale date for wash sale detection
            lot.sale_date = tax_date
                
            self.tax_events.append({
                "date": tax_date,
                "ticker": lot.identifier,
                "shares": lot.shares,
                "purchase_price": lot.purchase_price,
                "sale_price": current_price,
                "gain": gain,
                "holding_period_days": holding_period,
                "type": "long_term" if is_long_term else "short_term"
            })

    def calculate_tax_burden(self, sold_lots: List[TaxLot], current_price: float, 
                            short_term_rate: float = 0.35, long_term_rate: float = 0.15, 
                            tax_date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Calculate tax burden with different rates for short and long-term gains.
        """
        if tax_date is None:
            tax_date = datetime.now()
            
        short_term_gains = 0
        long_term_gains = 0
        
        for lot in sold_lots:
            gain = self._calculate_gain(lot, current_price)
            if self._is_long_term(lot, tax_date):
                long_term_gains += gain
            else:
                short_term_gains += gain
        
        tax_burden = (short_term_gains * short_term_rate) + (long_term_gains * long_term_rate)
        
        return {
            "short_term_gains": short_term_gains,
            "long_term_gains": long_term_gains,
            "short_term_tax": short_term_gains * short_term_rate,
            "long_term_tax": long_term_gains * long_term_rate,
            "total_tax_burden": tax_burden
        }
    
    def find_tax_loss_harvesting_opportunities(self, current_prices: Dict[str, float], 
                                              threshold: float = -1000) -> List[Dict[str, Any]]:
        """Find positions with unrealized losses that could be harvested."""
        opportunities = []
        
        for ticker, lots in self.tax_lots.items():
            if ticker not in current_prices:
                continue
                
            current_price = current_prices[ticker]
            
            for lot in lots:
                unrealized_gain = self._calculate_gain(lot, current_price)
                if unrealized_gain < threshold:
                    opportunities.append({
                        "ticker": ticker,
                        "shares": lot.shares,
                        "purchase_price": lot.purchase_price,
                        "current_price": current_price,
                        "unrealized_loss": unrealized_gain,
                        "purchase_date": lot.purchase_date
                    })
                    
        return opportunities
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Get a summary of the portfolio including unrealized gains/losses."""
        summary = {
            "total_value": 0,
            "cost_basis": 0,
            "unrealized_gains": 0,
            "positions": {},
            "realized_gains": {
                "short_term": self.realized_short_term_gains,
                "long_term": self.realized_long_term_gains,
                "total": self.realized_short_term_gains + self.realized_long_term_gains
            }
        }
        
        for ticker, lots in self.tax_lots.items():
            if ticker not in current_prices:
                continue
                
            current_price = current_prices[ticker]
            position_shares = sum(lot.shares for lot in lots)
            position_cost = sum(lot.shares * lot.purchase_price for lot in lots)
            position_value = position_shares * current_price
            position_gain = position_value - position_cost
            
            summary["positions"][ticker] = {
                "shares": position_shares,
                "cost_basis": position_cost,
                "current_value": position_value,
                "unrealized_gain": position_gain,
                "average_cost": position_cost / position_shares if position_shares > 0 else 0
            }
            
            summary["total_value"] += position_value
            summary["cost_basis"] += position_cost
            summary["unrealized_gains"] += position_gain
        
        return summary
    
    def check_wash_sales(self, sold_lot: TaxLot, purchase_date: datetime, ticker: str) -> bool:
        """
        Check if a new purchase would trigger a wash sale rule.
        Wash sale rule: Loss is disallowed if you buy the same security
        within 30 days before or after selling at a loss.
        """
        if sold_lot.identifier != ticker:
            return False
            
        if not hasattr(sold_lot, 'sale_date') or sold_lot.sale_date is None:
            return False
            
        days_difference = abs((purchase_date - sold_lot.sale_date).days)
        return days_difference <= 30