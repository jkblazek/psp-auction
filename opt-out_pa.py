import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

np.random.seed()

# Node class representing a buyer or seller
class Node:
    def __init__(self, id, price, quantity):
        self.id = id
        self.price = price
        self.quantity = quantity  # Positive for buyers, negative for sellers
        self.bid = np.random.uniform(50, 100)  # Initial bid for buyers
        self.influence_set = []  # Track influence from connected nodes

# Create a network where buyers are connected to multiple sellers
def create_random_market_network(buyers, sellers):
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

# Calculate time-windowed averages for influence sets
def update_influence_sets(G_seller_buyer, buyers, sellers, window_size=5):
    for buyer in buyers:
        # Track recent prices and supply from connected sellers
        influence_set = []
        for seller_id in G_seller_buyer.neighbors(buyer.id):
            seller = G_seller_buyer.nodes[seller_id]['obj']
            influence_set.append(seller.price)
        
        # Take a moving average (windowed influence set)
        if len(buyer.influence_set) >= window_size:
            buyer.influence_set.pop(0)  # Remove oldest entry
        buyer.influence_set.append(np.mean(influence_set) if influence_set else 0)

# Utility calculation with opt-out mechanism
def update_bids_with_progressive_opt_out(G_seller_buyer, buyers, sellers, buyer_config, seller_config, window_size=5):
    for buyer in buyers:
        connected_sellers = list(G_seller_buyer.neighbors(buyer.id))
        seller_prices = {seller_id: G_seller_buyer.nodes[seller_id]['obj'].price for seller_id in connected_sellers}
        seller_prices_sorted = sorted(seller_prices.items(), key=lambda x: x[1])

        demand_left = buyer.quantity
        total_utility = 0

        # Allocate demand progressively across sellers
        for seller_id, price in seller_prices_sorted:
            seller = G_seller_buyer.nodes[seller_id]['obj']
            allocation = min(seller.quantity, demand_left)
            utility = np.log(1 + max(0, allocation)) - price * allocation  # Fix log issue
            
            # Track total utility and allocate progressively
            total_utility += utility
            demand_left -= allocation
            
            if demand_left <= 0:  # Buyer is fully satisfied
                break

        # Check influence sets to determine if buyer opts out
        avg_influence = np.mean(buyer.influence_set) if buyer.influence_set else 0
        if total_utility > avg_influence:  # Opt-out condition
            for seller_id, _ in seller_prices_sorted:  # Extract only the seller_id
                G_seller_buyer.remove_edge(buyer.id, seller_id)  # Opt-out by removing edge
                break  # Opt-out from remaining sellers after utility threshold is met

# Track price history and plot
def track_price_history(price_history, buyer_price_history, buyers, sellers, iteration):
    for seller in sellers:
        if seller.id not in price_history:
            price_history[seller.id] = [None] * iteration  # Fill in past with None
        price_history[seller.id].append(seller.price)
    
    for buyer in buyers:
        if buyer.id not in buyer_price_history:
            buyer_price_history[buyer.id] = [None] * iteration  # Fill in past with None
        buyer_price_history[buyer.id].append(buyer.bid)

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

# Main simulation with opt-out
def main_network_simulation_with_opt_out():
    config = {
        "num_buyers": 20,
        "num_sellers": 5,
        "iterations": 20,
        "window_size": 5  # Influence set window size
    }

    # Seller and buyer configuration
    seller_config = {"seller_price_high": 100, "seller_price_low": 50}
    buyer_config = {"buyer_quantity_high": 20, "buyer_quantity_low": 10}

    # Initialize buyers and sellers
    sellers = [Node(f"Seller_{i}", price=np.random.uniform(50, 100), quantity=-np.random.uniform(10, 20)) for i in range(config["num_sellers"])]
    buyers = [Node(f"Buyer_{i}", price=np.random.uniform(20, 50), quantity=np.random.uniform(10, 20)) for i in range(config["num_buyers"])]

    # Create networks
    G_seller_buyer = create_random_market_network(buyers, sellers)
    G_buyer_buyer = create_random_market_network(buyers, sellers)

    price_history = {seller.id: [] for seller in sellers}
    buyer_price_history = {buyer.id: [] for buyer in buyers}

    # Run simulation
    for iteration in range(config["iterations"]):
        update_influence_sets(G_seller_buyer, buyers, sellers, window_size=config["window_size"])
        update_bids_with_progressive_opt_out(G_seller_buyer, buyers, sellers, buyer_config, seller_config, window_size=config["window_size"])
        track_price_history(price_history, buyer_price_history, buyers, sellers, iteration)

    # Visualize results
    seller_buyer_connections = {seller.id: list(G_seller_buyer.neighbors(seller.id)) for seller in sellers}
    plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections)

# Run the simulation
main_network_simulation_with_opt_out()