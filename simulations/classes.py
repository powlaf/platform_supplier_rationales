import random


class Supplier:
    _id_counter = 0

    def __init__(self, patience):
        self.id = self._get_next_id()
        self.cost = random.uniform(0, 1)
        self.patience = patience
        self.waiting_time = 0

    @classmethod
    def _get_next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def reset(cls):
        cls._id_counter = 0

class Customer:
    _id_counter = 0

    def __init__(self, patience):
        self.id = self._get_next_id()
        self.valuation = random.uniform(0, 1)
        self.patience = patience
        self.waiting_time = 0

    @classmethod
    def _get_next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def reset(cls):
        cls._id_counter = 0


class Match:

    def __init__(self, customer, supplier, price, commission):
        self.customer = customer
        self.supplier = supplier
        self.price = price

        self.status = True if (customer.valuation >= price) and (supplier.cost <= price * (1-commission)) else False

        if self.status:
            self.welfare_supplier = price * (1 - commission) - supplier.cost
            self.welfare_customer = customer.valuation - price
            self.welfare_platform = price * commission
            self.welfare = self.welfare_customer + self.welfare_supplier + self.welfare_platform
        else:
            self.welfare_supplier, self.welfare_customer, self.welfare_platform, self.welfare = 0, 0, 0, 0

