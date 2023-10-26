# This code is adapted from https://github.com/microservices-demo/load-test

import base64
from locust import HttpUser, task, constant_throughput
from random import choice


# Define different types of users as:
# 1. A user does not log in,
#    and browses front-end, catalogue, and an item (under catalogue).
#    Requests per user:
#      1 * front-end,
#      2 * catalogue
#      Total: 3
class AnonymousUser(HttpUser):
    weight = 6
    # The task runs (at most) X times per second.
    wait_time = constant_throughput(0.6)

    @task
    def load(self):
        # item_id "03fef6ac-1896-4ce8-bd69-b798f85c6e0b"
        # can cause HTTPError('406 Client Error: Not Acceptable for url: /orders').
        # Because the cost of the item exceeds the credit card limit.
        item_ids = ["3395a43e-2d88-40de-b95f-e00e1502085b",
                   "510a0d7e-8e83-4193-b483-e27e09ddc34d",
                   "808a2de1-1aaa-4c25-a9b9-6612e8f29a38",
                   "819e1fbf-8b7e-4f6d-811f-693534916a8b",
                   "837ab141-399e-4c1f-9abc-bace40296bac",
                   "a0a4f044-b040-410d-8ead-4de0446aec7e",
                   "d3588630-ad8e-49df-bbd7-3167f7efb246",
                   "zzz4f044-b040-410d-8ead-4de0446aec7e"]        
        item_id = choice(item_ids)
        
        # Request front-end.
        self.client.get("/index.html", name="front-end")
        # Request catalogue.
        self.client.get("/category.html", name="catalogue")
        # Request catalogue.
        self.client.get(f"/detail.html?id={item_id}", name="catalogue")


# 2. A user browses front-end,
#    logs in,
#    browses catalogue and an item (under catalogue),
#    and adds this item to the cart.
#    Requests per user:
#      1 * front-end,
#      1 * user,
#      2 * catalogue,
#      2 * carts
#      Total: 6
class ShoppingUser(HttpUser):
    weight = 3
    # The task runs (at most) X times per second.
    wait_time = constant_throughput(0.3)

    @task
    def load(self):
        # Use a Base64-encoded string as the credentials
        # in an HTTP Basic Authentication header.
        base64string = base64.b64encode(b'user:password').decode('utf-8').replace('\n', '')

        # item_id "03fef6ac-1896-4ce8-bd69-b798f85c6e0b"
        # can cause HTTPError('406 Client Error: Not Acceptable for url: /orders').
        # Because the cost of the item exceeds the credit card limit.
        item_ids = ["3395a43e-2d88-40de-b95f-e00e1502085b",
                   "510a0d7e-8e83-4193-b483-e27e09ddc34d",
                   "808a2de1-1aaa-4c25-a9b9-6612e8f29a38",
                   "819e1fbf-8b7e-4f6d-811f-693534916a8b",
                   "837ab141-399e-4c1f-9abc-bace40296bac",
                   "a0a4f044-b040-410d-8ead-4de0446aec7e",
                   "d3588630-ad8e-49df-bbd7-3167f7efb246",
                   "zzz4f044-b040-410d-8ead-4de0446aec7e"]        
        item_id = choice(item_ids)
        
        # Request front-end.
        self.client.get("/index.html", name="front-end")
        # Request user.
        self.client.get("/login", headers={"Authorization": "Basic %s" % base64string}, name="user")
        # Request catalogue.
        self.client.get("/category.html", name="catalogue")
        # Request catalogue.
        self.client.get(f"/detail.html?id={item_id}", name="catalogue")
        # Request carts.
        self.client.delete("/cart", name="carts")
        # Request carts.
        self.client.post("/cart", json={"id": item_id, "quantity": 1}, name="carts")


# 3. A user browses front-end,
#    logs in,
#    browses catalogue and an item (under catalogue),
#    adds this item to the cart,
#    goes to the basket (under carts),
#    and places the order.
#    Requests per user:
#      1 * front-end,
#      1 * user,
#      2 * catalogue,
#      3 * carts,
#      1 * orders,
#      1 * payment,
#      1 * shipping
#      Total: 10
class PayingUser(HttpUser):
    weight = 1
    # The task runs (at most) X times per second.
    wait_time = constant_throughput(0.1)

    @task
    def load(self):
        # Use a Base64-encoded string as the credentials
        # in an HTTP Basic Authentication header.
        base64string = base64.b64encode(b'Eve_Berger:eve').decode('utf-8').replace('\n', '')

        # item_id "03fef6ac-1896-4ce8-bd69-b798f85c6e0b"
        # can cause HTTPError('406 Client Error: Not Acceptable for url: /orders').
        # Because the cost of the item exceeds the credit card limit.
        item_ids = ["3395a43e-2d88-40de-b95f-e00e1502085b",
                   "510a0d7e-8e83-4193-b483-e27e09ddc34d",
                   "808a2de1-1aaa-4c25-a9b9-6612e8f29a38",
                   "819e1fbf-8b7e-4f6d-811f-693534916a8b",
                   "837ab141-399e-4c1f-9abc-bace40296bac",
                   "a0a4f044-b040-410d-8ead-4de0446aec7e",
                   "d3588630-ad8e-49df-bbd7-3167f7efb246",
                   "zzz4f044-b040-410d-8ead-4de0446aec7e"]        
        item_id = choice(item_ids)
        
        # Request front-end.
        self.client.get("/index.html", name="front-end")
        # Request user.
        self.client.get("/login", headers={"Authorization": "Basic %s" % base64string}, name="user")
        # Request catalogue.
        self.client.get("/category.html", name="catalogue")
        # Request catalogue.
        self.client.get(f"/detail.html?id={item_id}", name="catalogue")
        # Request carts.
        self.client.delete("/cart", name="carts")
        # Request carts.
        self.client.post("/cart", json={"id": item_id, "quantity": 1}, name="carts")
        # Request carts.
        self.client.get("/basket.html", name="carts")
        # Request orders, payment, and shipping, one of each.
        self.client.post("/orders", name="orders")
