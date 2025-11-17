import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

# Node class representing a buyer or seller
class Node:
    def __init__(self, id, price, quantity):
        self.id = id
        self.price = price
        self.quantity = quantity  # Positive for buyers, negative for sellers
        self.bid = np.random.uniform(50, 100)  # Initial bid for buyers

# Create a network where buyers are connected to multiple sellers
def create_seller_shared_buyer_network(buyers, sellers):
    G_seller_buyer = nx.Graph()
    G_buyer_buyer = nx.Graph()

    for seller in sellers:
        G_seller_buyer.add_node(seller.id, obj=seller)
    
    for buyer in buyers:
        G_seller_buyer.add_node(buyer.id, obj=buyer)
        G_buyer_buyer.add_node(buyer.id, obj=buyer)

    # Randomly connect buyers to sellers
    for seller in sellers:
        connected_buyers = random.sample(buyers, k=random.randint(1, len(buyers)))
        for buyer in connected_buyers:
            G_seller_buyer.add_edge(seller.id, buyer.id)

    # Connect buyers to each other based on shared sellers
    for seller in sellers:
        connected_buyers = [buyer for buyer in G_seller_buyer.neighbors(seller.id)]
        for i in range(len(connected_buyers)):
            for j in range(i + 1, len(connected_buyers)):
                G_buyer_buyer.add_edge(connected_buyers[i], connected_buyers[j])

    return G_seller_buyer, G_buyer_buyer

# Apply PSP logic to adjust seller prices and buyer bids
def update_bids_psp(G_seller_buyer, G_buyer_buyer):
    for seller_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[seller_id]['obj']
        
        # Seller logic: Adjust based on connected buyer bids
        if "Seller" in seller_id:  # Seller
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(seller_id)]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]  # Progressive Second Price: Second-highest bid
                node.price = second_highest_bid  # Sellers price at second-highest buyer bid
            
        # Buyer logic: Adjust based on connected sellers and other buyers
        elif "Buyer" in seller_id:  # Buyer
            seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(seller_id) if "Seller" in neighbor]
            if seller_prices:
                min_seller_price = min(seller_prices)  # Buyers prefer the lowest price
                other_buyer_bids = [G_buyer_buyer.nodes[neighbor]['obj'].bid for neighbor in G_buyer_buyer.neighbors(seller_id)]
                if other_buyer_bids:
                    avg_buyer_influence = np.mean(other_buyer_bids)  # Influence from other buyers
                    # Adjust buyer bid: 50% based on seller prices, 50% based on other buyers
                    G_seller_buyer.nodes[seller_id]['obj'].bid = (avg_buyer_influence + min_seller_price) / 2

# Calculate Supply-Demand Ratio (SDR)
def calculate_sdr(buyers, sellers):
    total_supply = abs(sum(seller.quantity for seller in sellers))
    total_demand = sum(buyer.quantity for buyer in buyers)

    if total_demand == 0:
        return float('inf')
    return total_supply / total_demand

# Adjust market participants based on SDR
def adjust_market_participants(SDR, buyers, sellers, SDR_threshold_high, SDR_threshold_low):
    if SDR > SDR_threshold_high:  # Too much supply
        if len(sellers) > 0:
            sellers.pop()  # Remove a random seller
        if random.random() < 0.5:
            buyers.append(Node(f"Buyer_{len(buyers)}", price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20)))

    elif SDR < SDR_threshold_low:  # Too much demand
        if len(buyers) > 0:
            buyers.pop()  # Remove a random buyer
        if random.random() < 0.5:
            sellers.append(Node(f"Seller_{len(sellers)}", price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20)))

# OU Process for adjusting SDR
def ou_process_sdr(SDR, mu, theta, sigma, dt):
    dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
    SDR += theta * (mu - SDR) * dt + sigma * dW
    return SDR

# Plot the network and price history (including buyer price adjustments)
def plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections):
    pos = nx.spring_layout(G_seller_buyer)

    plt.figure(figsize=(12, 10))

    # Basic colors for the 5 sellers in the legend
    basic_colors = ['red', 'blue', 'green', 'purple', 'orange']
    # Full spectrum for actual plot color coding
    full_spectrum_colors = plt.cm.rainbow(np.linspace(0, 1, len(seller_buyer_connections)))

    # First subplot: Combined seller-buyer network with seller clusters
    plt.subplot(2, 1, 1)
    for idx, (seller_id, buyers) in enumerate(seller_buyer_connections.items()):
        color = full_spectrum_colors[idx]  # Use the full spectrum for clusters
        # Draw the seller node in the cluster color
        nx.draw_networkx_nodes(G_seller_buyer, pos, nodelist=[seller_id], node_color=[color], node_size=500, label=f"Seller {seller_id}")
        # Draw the buyer nodes in the same cluster color
        buyer_ids = buyers  # Now correct, buyers are already node IDs
        nx.draw_networkx_nodes(G_seller_buyer, pos, nodelist=buyer_ids, node_color=[color] * len(buyer_ids), node_size=500)
        # Draw seller-buyer edges in the cluster color
        for buyer_id in buyer_ids:
            nx.draw_networkx_edges(G_seller_buyer, pos, edgelist=[(seller_id, buyer_id)], edge_color=[color], style='solid', width=2)

    # Draw the buyer-buyer network (dotted edges)
    dotted_edges = [(u, v) for u, v in G_buyer_buyer.edges]
    nx.draw_networkx_edges(G_buyer_buyer, pos, edgelist=dotted_edges, style='dotted', edge_color='blue')

    # Add labels and title
    nx.draw_networkx_labels(G_seller_buyer, pos)
    plt.title("Seller-Buyer and Buyer-Buyer Network")

    # Second subplot: Plot seller price adjustments and buyer price adjustments over time
    plt.subplot(2, 1, 2)
    for idx, (seller_id, buyers) in enumerate(seller_buyer_connections.items()):
        color = full_spectrum_colors[idx]  # Use full spectrum for plot
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

# Main function for PSP logic, OU process, and market entry/exit with buyer price tracking
def main_psp_ou_with_entry_exit_and_buyer_prices(iterations=100):
    num_buyers = 10
    num_sellers = 3

    # Initialize buyers and sellers
    buyers = [Node(f"Buyer_{i}", price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20)) for i in range(num_buyers)]
    sellers = [Node(f"Seller_{i}", price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20)) for i in range(num_sellers)]

    SDR = 1.0  # Initial SDR
    mu = 1.0  # Equilibrium SDR
    theta = 0.1  # Speed of adjustment
    sigma = 0.05  # Volatility
    dt = 1

    SDR_threshold_high = 1.5  # Too much supply
    SDR_threshold_low = 0.7  # Too much demand

    price_history = {seller.id: [] for seller in sellers}
    buyer_price_history = {buyer.id: [] for buyer in buyers}

    for iteration in range(iterations):
        # Calculate SDR
        SDR = calculate_sdr(buyers, sellers)

        # Adjust SDR using OU process
        SDR = ou_process_sdr(SDR, mu, theta, sigma, dt)

        # Adjust market participants based on SDR
        adjust_market_participants(SDR, buyers, sellers, SDR_threshold_high, SDR_threshold_low)

        # Ensure price history for any new sellers and buyers
        for seller in sellers:
            if seller.id not in price_history:
                price_history[seller.id] = []
        for buyer in buyers:
            if buyer.id not in buyer_price_history:
                buyer_price_history[buyer.id] = []

        # Create new networks after adjustments
        G_seller_buyer, G_buyer_buyer = create_seller_shared_buyer_network(buyers, sellers)

        # Apply PSP logic
        update_bids_psp(G_seller_buyer, G_buyer_buyer)

        # Track seller and buyer prices over time
        for seller in sellers:
            price_history[seller.id].append(seller.price)
        for buyer in buyers:
            buyer_price_history[buyer.id].append(buyer.bid)

    # Extract seller-buyer connections
    seller_buyer_connections = {}
    for seller in sellers:
        seller_buyer_connections[seller.id] = [buyer for buyer in G_seller_buyer.neighbors(seller.id) if 'Buyer' in buyer]

    # Plot network and price history with buyer adjustments
    plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections)

# Run the simulation
main_psp_ou_with_entry_exit_and_buyer_prices(iterations=100)  # Running for 100 iterations