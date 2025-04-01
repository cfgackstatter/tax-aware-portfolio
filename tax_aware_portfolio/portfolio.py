from collections import deque
from tax_aware_portfolio.tax_lot import TaxLot

class Portfolio:
    def __init__(self):
        self.tax_lots = {}

    def buy(self, identifier, shares, purchase_price, purchase_date):
        if identifier not in self.tax_lots:
            self.tax_lots[identifier] = deque()

        self.tax_lots[identifier].append(TaxLot(identifier, shares, purchase_price, purchase_date))

    def sell(self, identifier, shares_to_sell, method='FIFO'):
        if identifier not in self.tax_lots or sum(lot.shares for lot in self.tax_lots[identifier]) < shares_to_sell:
            raise ValueError("Insufficient shares to sell")
        
        sold_lots = []
        remaining_shares = shares_to_sell

        if method == 'FIFO':
            while remaining_shares > 0:
                lot = self.tax_lots[identifier][0]
                if lot.shares <= remaining_shares:
                    sold_lots.append(self.tax_lots[identifier].popleft())
                    remaining_shares -= lot.shares
                else:
                    lot.shares -= remaining_shares
                    sold_lots.append(TaxLot(lot.identifier, remaining_shares, lot.purchase_price, lot.purchase_date))
                    remaining_shares = 0
        elif method == 'HIFO':
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

        return sold_lots
    
    def calculate_tax_burden(self, sold_lots, current_price):
        total_taxable_gain = 0
        for lot in sold_lots:
            gain = (current_price - lot.purchase_price) * lot.shares
            total_taxable_gain += gain
        return total_taxable_gain
    
    def __repr__(self):
        return f"Portfolio({self.tax_lots})"