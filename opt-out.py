import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import deque

# Node class representing a buyer or seller
class Node:
    def __init__(self, id, price, quantity, gamma=1.0):
        self.id = id
        self.price = price
        self.quantity = quantity  # Positive for buyers, negative for sellers
        self.bid = np.random.uniform(50, 100)  # Initial bid for buyers
        self.gamma = gamma  # Elasticity parameter for buyers
        self.valuation_function = lambda z: self.gamma * np.log(1 + z)

# Create a network where buyers are connected to multiple sellers
def create_random_market_network(buyers, sellers):
    print("Creating random network")
    G_seller_buyer = nx.Graph()

    for seller in sellers:
        G_seller_buyer.add_node(seller.id, obj=seller)
    for buyer in buyers:
        G_seller_buyer.add_node(buyer.id, obj=buyer)

    # Randomly connect buyers to sellers
    for seller in sellers:
        connected_buyers = random.sample(buyers, k=random.randint(1, len(buyers)))
        for buyer in connected_buyers:
            G_seller_buyer.add_edge(seller.id, buyer.id)

    return G_seller_buyer

# Create buyer-buyer network based on shared sellers
def create_buyer_buyer_network(buyers, sellers, G_seller_buyer):
    print("Creating buyer-buyer network based on shared sellers")
    G_buyer_buyer = nx.Graph()
    for buyer in buyers:
        G_buyer_buyer.add_node(buyer.id, obj=buyer)

    for seller in sellers:
        connected_buyers = [buyer for buyer in G_seller_buyer.neighbors(seller.id)]
        for i in range(len(connected_buyers)):
            for j in range(i + 1, len(connected_buyers)):
                G_buyer_buyer.add_edge(connected_buyers[i], connected_buyers[j])

    return G_buyer_buyer

# Utility calculation for buyers
def calculate_utility(buyer, allocation, price):
    valuation = buyer.valuation_function(allocation)
    cost = allocation * price
    return valuation - cost

# Opt-out function based on utility maximization
def opt_out_utility_based(buyer, connected_sellers, G_seller_buyer, gamma=1.0):
    sellers_sorted = sorted(connected_sellers, key=lambda s: s.price)
    total_demand = buyer.quantity
    total_allocated = 0
    chosen_sellers = []

    for seller in sellers_sorted:
        allocation = min(abs(seller.quantity), total_demand - total_allocated)
        price = seller.price
        utility = calculate_utility(buyer, allocation, price)
        if utility > 0:
            chosen_sellers.append(seller)
            total_allocated += allocation
        if total_allocated >= total_demand:
            break

    return chosen_sellers

# Update bids with opt-out mechanism
def update_bids_with_utility_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config):
    for node_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[node_id]['obj']

        if "Buyer" in node_id:
            connected_sellers = [G_seller_buyer.nodes[neighbor]['obj'] for neighbor in G_seller_buyer.neighbors(node_id) if "Seller" in neighbor]

            if len(connected_sellers) > 2:
                chosen_sellers = opt_out_utility_based(node, connected_sellers, G_seller_buyer)

                for seller in chosen_sellers:
                    G_seller_buyer.add_edge(node_id, seller.id)
                    node_bid = node.bid
                    seller_price = seller.price
                    node.bid = (node_bid + seller_price) / 2

            else:
                seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(node_id) if "Seller" in neighbor]
                if seller_prices:
                    min_seller_price = min(seller_prices)
                    other_buyer_bids = [G_buyer_buyer.nodes[neighbor]['obj'].bid for neighbor in G_buyer_buyer.neighbors(node_id)]
                    second_highest_buyer_bid = sorted(other_buyer_bids)[-2] if len(other_buyer_bids) > 1 else other_buyer_bids[0]
                    node.bid = (min_seller_price + second_highest_buyer_bid) / 2

        elif "Seller" in node_id:
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(node_id) if "Buyer" in neighbor]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]
                node.price = second_highest_bid

# Plot network and price history
def plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections):
    pos = nx.spring_layout(G_seller_buyer)

    plt.figure(figsize=(12, 10))

    # Full spectrum for actual plot color coding
    full_spectrum_colors = plt.cm.rainbow(np.linspace(0, 1, len(seller_buyer_connections)))

    # First subplot: Combined seller-buyer network with seller clusters
    plt.subplot(2, 1, 1)
    for idx, (seller_id, buyers) in enumerate(seller_buyer_connections.items()):
        color = full_spectrum_colors[idx]
        # Draw the seller node in the cluster color
        nx.draw_networkx_nodes(G_seller_buyer, pos, nodelist=[seller_id], node_color=[color], node_size=500, label=f"Seller {seller_id}")
        # Draw the buyer nodes in the same cluster color
        buyer_ids = buyers
        nx.draw_networkx_nodes(G_seller_buyer, pos, nodelist=buyer_ids, node_color=[color] * len(buyer_ids), node_size=500)
        # Draw seller-buyer edges in the cluster color
        for buyer_id in buyer_ids:
            nx.draw_networkx_edges(G_seller_buyer, pos, edgelist=[(seller_id, buyer_id)], edge_color=[color], style='solid', width=2)

    # Draw the buyer-buyer network (dotted edges)
    dotted_edges = [(u, v) for u, v in G_buyer_buyer.edges]
    nx.draw_networkx_edges(G_buyer_buyer, pos, edgelist=dotted_edges, style='dotted', edge_color='blue')

    # Add labels and title
    nx.draw_networkx_labels(G_seller_buyer, pos)
    plt.title("Buyer-Seller Network with Buyer-Buyer Connections")

    # Second subplot: Plot seller price adjustments and buyer price adjustments over time
    plt.subplot(2, 1, 2)
    for idx, (seller_id, buyers) in enumerate(seller_buyer_connections.items()):
        color = full_spectrum_colors[idx]
        # Plot seller's price as a solid line
        plt.plot(np.arange(len(price_history[seller_id])), price_history[seller_id], linestyle='solid', color=color, label=f"Seller {seller_id}")
        # Plot each buyer's price as a dotted line
        for buyer in buyers:
            plt.plot(np.arange(len(buyer_price_history[buyer])), buyer_price_history[buyer], linestyle='dotted', color=color)

    plt.title("Price Adjustments Over Time (Sellers and Buyers)")
    plt.xlabel("Iterations")
    plt.ylabel("Price")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Simulation setup
def main_network_simulation_with_opt_out():
    config = {
        "network_type": "random",
        "num_buyers": 20,
        "num_sellers": 5,
        "iterations": 20,
        "gamma": 1.0  # Elasticity parameter for buyer valuation
    }

    seller_config = {
        "seller_price_high": 100,
        "seller_price_low": 50,
        "seller_quantity_high": 20,
        "seller_quantity_low": 10
    }
    buyer_config = {
        "buyer_price_high": 50,
        "buyer_price_low": 20,
        "buyer_quantity_high": 20,
        "buyer_quantity_low": 10
    }

    sellers = [Node(f"Seller_{i}", price=np.random.uniform(seller_config["seller_price_low"], seller_config["seller_price_high"]),
                    quantity=-np.random.uniform(seller_config["seller_quantity_low"], seller_config["seller_quantity_high"])) for i in range(config["num_sellers"])]

    buyers = [Node(f"Buyer_{i}", price=np.random.uniform(buyer_config["buyer_price_low"], buyer_config["buyer_price_high"]),
                   quantity=np.random.uniform(buyer_config["buyer_quantity_low"], buyer_config["buyer_quantity_high"]), gamma=config["gamma"]) for i in range(config["num_buyers"])]

    G_seller_buyer = create_random_market_network(buyers, sellers)
    G_buyer_buyer = create_buyer_buyer_network(buyers, sellers, G_seller_buyer)
    G_seller_seller = nx.Graph()  # This can be used if you want to implement seller-seller connections

    price_history = {seller.id: [] for seller in sellers}
    buyer_price_history = {buyer.id: [] for buyer in buyers}

    for iteration in range(config["iterations"]):
        print(f"Iteration {iteration}")
        update_bids_with_utility_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config)

        # Track price history
        for seller in sellers:
            price_history[seller.id].append(seller.price)

        for buyer in buyers:
            buyer_price_history[buyer.id].append(buyer.bid)

    # Visualize the results
    seller_buyer_connections = {seller.id: list(G_seller_buyer.neighbors(seller.id)) for seller in sellers}
    plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections)

    print("Simulation completed!")

# Run the simulation
if __name__ == "__main__":
    main_network_simulation_with_opt_out()