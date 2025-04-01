from datetime import datetime

class TaxLot:
    def __init__(self, identifier, shares, purchase_price, purchase_date):
        self.identifier = identifier
        self.shares = shares
        self.purchase_price = purchase_price
        self.purchase_date = purchase_date

    def __repr__(self):
        return f"TaxLot({self.identifier}, {self.shares}, {self.purchase_price}, {self.purchase_date})"