import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

class nxNode(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = None  # Price (float)
        self.q = None  # Quantity (int)

    def set_node_attributes(self, p: float, q: int):
        self.p = p
        self.q = q

    def get_node_attributes(self):
        return self.p, self.q


class PSPAuction:
    def __init__(self, num_buyers, num_sellers):
        self.network = nxNode()
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.create_small_world_network()

    def create_small_world_network(self):
        # Create separate buyers and sellers
        buyers = range(self.num_buyers)
        sellers = range(self.num_buyers, self.num_buyers + self.num_sellers)

        # Add nodes with initial price and quantity (negative for sellers, positive for buyers)
        for buyer in buyers:
            self.network.add_node(buyer)
            self.network.nodes[buyer]["flash_memory"] = 0  # Initialize flash memory for each buyer
            self.network.nodes[buyer]["type"] = "buyer"
            # Positive demand for buyers
            self.network.nodes[buyer]["quantity"] = random.randint(1, 10)
            self.network.nodes[buyer]["price"] = random.uniform(50, 100)

        for seller in sellers:
            self.network.add_node(seller)
            self.network.nodes[seller]["type"] = "seller"
            # Negative demand for sellers (supply)
            self.network.nodes[seller]["quantity"] = random.randint(-10, -1)
            self.network.nodes[seller]["price"] = random.uniform(50, 100)

        # Connect buyers and sellers
        for seller in sellers:
            connected_buyers = random.sample(buyers, random.randint(2, 4))  # Randomly connect to 2-4 buyers
            for buyer in connected_buyers:
                self.network.add_edge(buyer, seller, connection_type='solid')

                # Connect buyers who share the same seller (dotted lines)
                for other_buyer in connected_buyers:
                    if buyer != other_buyer and not self.network.has_edge(buyer, other_buyer):
                        self.network.add_edge(buyer, other_buyer, connection_type='dotted')

    def mean_reverting_supply_demand_ratio(self):
        # Generate a mean-reverting process for the ratio of supply to demand
        mu = 1.0  # Mean ratio of supply to demand
        theta = 0.5  # Rate of mean reversion
        sigma = 0.1  # Volatility

        ratio = mu
        for _ in range(100):
            ratio += theta * (mu - ratio) + sigma * np.random.randn()
        return ratio

    def adjust_bids(self, buyer):
        connected_sellers = [n for n in self.network.neighbors(buyer) if self.network.nodes[n]["type"] == "seller"]
        connected_buyers = [n for n in self.network.neighbors(buyer) if self.network.nodes[n]["type"] == "buyer"]

        for seller in connected_sellers:
            seller_price = self.network.nodes[seller]["price"]

            # Influence from the connected buyers
            buyer_influence = np.mean([self.network.nodes[buyer]["flash_memory"] for buyer in connected_buyers]) if connected_buyers else 0

            # Update buyer's price based on flash memory and influence from connected buyers
            new_bid = seller_price + buyer_influence + np.random.uniform(-5, 5)  # Random noise in bidding

            # Store bid adjustment in flash memory
            self.network.nodes[buyer]["flash_memory"] = new_bid - self.network.nodes[buyer]["price"]

            # Update buyer's bid
            self.network.nodes[buyer]["price"] = new_bid

    def run_auction(self, rounds=10):
        for _ in range(rounds):
            for buyer in [n for n in self.network.nodes if self.network.nodes[n]["type"] == "buyer"]:
                self.adjust_bids(buyer)

    def plot_network(self):
        pos = nx.spring_layout(self.network)

        # Draw the nodes
        nx.draw_networkx_nodes(self.network, pos, nodelist=[n for n in self.network.nodes if self.network.nodes[n]["type"] == "buyer"], node_color='blue', node_size=500, label='Buyers')
        nx.draw_networkx_nodes(self.network, pos, nodelist=[n for n in self.network.nodes if self.network.nodes[n]["type"] == "seller"], node_color='red', node_size=500, label='Sellers')

        # Draw the edges with solid and dotted lines
        solid_edges = [(u, v) for u, v, d in self.network.edges(data=True) if d['connection_type'] == 'solid']
        dotted_edges = [(u, v) for u, v, d in self.network.edges(data=True) if d['connection_type'] == 'dotted']

        nx.draw_networkx_edges(self.network, pos, edgelist=solid_edges, style='solid')
        nx.draw_networkx_edges(self.network, pos, edgelist=dotted_edges, style='dotted')

        # Labels for nodes
        labels = {n: f"{n}\nQ: {self.network.nodes[n]['quantity']}\nP: {round(self.network.nodes[n]['price'], 2)}" for n in self.network.nodes}
        nx.draw_networkx_labels(self.network, pos, labels=labels)

        plt.legend(scatterpoints=1)
        plt.title("PSP Auction Network")
        plt.show()


# Example Usage
auction = PSPAuction(num_buyers=5, num_sellers=3)
auction.run_auction(rounds=10)
auction.plot_network()