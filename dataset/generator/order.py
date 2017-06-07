# -*- coding: utf-8 -*-


class Order(object):

    @staticmethod
    def get_dataframe_header():
        return ['datetime', 'client_id', 'product_id', 'requested_qty', 'received_qty']

    def __init__(self, timestamp, client_id, product_id, requested_qty, received_qty):
        self.timestamp = timestamp
        self.client_id = client_id
        self.product_id = product_id
        self.requested_quantity = requested_qty
        self.received_quantity = received_qty

    def __str__(self):
        string = "Order: <Time: %d Client: %d Product: %d Requested: %f Received: %f>" % (self.timestamp, self.client_id, self.product_id, self.requested_quantity, self.received_quantity)
        return string

        # return "< Time:", self.timestamp, "Client:", self.client_id, "Product:",self.product_id, \
        #      "Requested:", self.requested_quantity, "Received:", self.received_quantity,">"

    def to_dict(self):
        return {
            'datetime': self.timestamp,
            'client_id': self.client_id,
            'product_id': self.product_id,
            'requested_qty': self.requested_quantity,
            'received_qty': self.received_quantity
        }