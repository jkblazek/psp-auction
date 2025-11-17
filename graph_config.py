import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# Function to load configuration from JSON file
def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to save a NetworkX graph to a GraphML file
def save_network(graph, filename):
    nx.write_graphml(graph, filename)

# Function to load a NetworkX graph from a GraphML file
def load_network(filename):
    return nx.read_graphml(filename)

# Node class representing a buyer or seller
class Node:
    def __init__(self, id, price, quantity):
        self.id = id
        self.price = price
        self.quantity = quantity
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

    for seller in sellers:
        connected_buyers = random.sample(buyers, k=random.randint(1, len(buyers)))
        for buyer in connected_buyers:
            G_seller_buyer.add_edge(seller.id, buyer.id)

    for seller in sellers:
        connected_buyers = [buyer for buyer in G_seller_buyer.neighbors(seller.id)]
        for i in range(len(connected_buyers)):
            for j in range(i + 1, len(connected_buyers)):
                G_buyer_buyer.add_edge(connected_buyers[i], connected_buyers[j])

    return G_seller_buyer, G_buyer_buyer

# Create seller-seller network based on shared buyers
def create_seller_seller_network(buyers, sellers, G_seller_buyer):
    G_seller_seller = nx.Graph()

    for seller in sellers:
        G_seller_seller.add_node(seller.id, obj=seller)
    
    for i, seller_1 in enumerate(sellers):
        for seller_2 in sellers[i+1:]:
            buyers_seller_1 = set(G_seller_buyer.neighbors(seller_1.id))
            buyers_seller_2 = set(G_seller_buyer.neighbors(seller_2.id))
            shared_buyers = buyers_seller_1.intersection(buyers_seller_2)
            
            if shared_buyers:
                G_seller_seller.add_edge(seller_1.id, seller_2.id, shared_buyers=list(shared_buyers))

    return G_seller_seller

# Update the price logic to include influence from the seller-seller network
def update_bids_psp(G_seller_buyer, G_buyer_buyer, G_seller_seller):
    for seller_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[seller_id]['obj']
        
        if "Seller" in seller_id:
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(seller_id)]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]
                node.price = second_highest_bid

            connected_seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_seller.neighbors(seller_id)]
            if connected_seller_prices:
                avg_seller_price = np.mean(connected_seller_prices)
                node.price = (node.price + avg_seller_price) / 2

        elif "Buyer" in seller_id:
            seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(seller_id) if "Seller" in neighbor]
            if seller_prices:
                min_seller_price = min(seller_prices)
                other_buyer_bids = [G_buyer_buyer.nodes[neighbor]['obj'].bid for neighbor in G_buyer_buyer.neighbors(seller_id)]
                if other_buyer_bids:
                    second_highest_buyer_bid = sorted(other_buyer_bids)[-2] if len(other_buyer_bids) > 1 else other_buyer_bids[0]
                    G_seller_buyer.nodes[seller_id]['obj'].bid = (min_seller_price + second_highest_buyer_bid) / 2

# Plot the network and price history
def plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections):
    pos = nx.spring_layout(G_seller_buyer)

    plt.figure(figsize=(12, 10))

    basic_colors = ['red', 'blue', 'green', 'purple', 'orange']
    full_spectrum_colors = plt.cm.rainbow(np.linspace(0, 1, len(seller_buyer_connections)))

    plt.subplot(2, 1, 1)
    for idx, (seller_id, buyers) in enumerate(seller_buyer_connections.items()):
        color = full_spectrum_colors[idx]
        nx.draw_networkx_nodes(G_seller_buyer, pos, nodelist=[seller_id], node_color=[color], node_size=500, label=f"Seller {seller_id}")
        buyer_ids = buyers
        nx.draw_networkx_nodes(G_seller_buyer, pos, nodelist=buyer_ids, node_color=[color] * len(buyer_ids), node_size=500)
        for buyer_id in buyer_ids:
            nx.draw_networkx_edges(G_seller_buyer, pos, edgelist=[(seller_id, buyer_id)], edge_color=[color], style='solid', width=2)

    dotted_edges = [(u, v) for u, v in G_buyer_buyer.edges]
    nx.draw_networkx_edges(G_buyer_buyer, pos, edgelist=dotted_edges, style='dotted', edge_color='blue')

    nx.draw_networkx_labels(G_seller_buyer, pos)
    plt.title("Buyer-Seller Network with Buyer-Buyer Connections")

    plt.subplot(2, 1, 2)
    for idx, (seller_id, buyers) in enumerate(seller_buyer_connections.items()):
        color = full_spectrum_colors[idx]
        plt.plot(np.arange(len(price_history[seller_id])), price_history[seller_id], linestyle='solid', color=color, label=f"Seller {seller_id}")
        for buyer in buyers:
            plt.plot(np.arange(len(buyer_price_history[buyer])), buyer_price_history[buyer], linestyle='dotted', color=color)

    plt.title("Price Adjustments Over Time (Sellers and Buyers)")
    plt.xlabel("Iterations")
    plt.ylabel("Price")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Calculate Supply-Demand Ratio (SDR)
def calculate_sdr(buyers, sellers):
    total_supply = abs(sum(seller.quantity for seller in sellers))
    total_demand = sum(buyer.quantity for buyer in buyers)
    if total_demand == 0:
        return float('inf')
    return total_supply / total_demand

# OU Process for adjusting SDR
def ou_process_sdr(SDR, mu, theta, sigma, dt):
    dW = np.random.normal(0, np.sqrt(dt))
    SDR += theta * (mu - SDR) * dt + sigma * dW
    return SDR

# Adjust market participants based on SDR
def adjust_market_participants(SDR, buyers, sellers, SDR_threshold_high, SDR_threshold_low):
    if SDR > SDR_threshold_high:
        if len(sellers) > 0:
            removed_seller = sellers.pop()
        if random.random() < 0.5:
            new_buyer = Node(f"Buyer_{len(buyers)}", price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20))
            buyers.append(new_buyer)

    elif SDR < SDR_threshold_low:
        if len(buyers) > 0:
            removed_buyer = buyers.pop()
        if random.random() < 0.5:
            new_seller = Node(f"Seller_{len(sellers)}", price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20))
            sellers.append(new_seller)

# Main function to load parameters from a file and save networks after simulation
def main_network_simulation_from_file(config_file):
    config = load_config(config_file)

    network_type = config["network_type"]
    num_buyers = config["num_buyers"]
    num_sellers = config["num_sellers"]
    SDR = config["SDR"]
    mu = config["mu"]
    theta = config["theta"]
    sigma = config["sigma"]
    iterations = config["iterations"]
    num_clusters = config.get("num_clusters", 2)

    buyers = [Node(f"Buyer_{i}", price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20)) for i in range(num_buyers)]
    sellers = [Node(f"Seller_{i}", price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20)) for i in range(num_sellers)]
    
    price_history = {seller.id: [] for seller in sellers}
    buyer_price_history = {buyer.id: [] for buyer in buyers}

    dt = 1

    SDR_threshold_high = 1.5
    SDR_threshold_low = 0.7

    # Create the selected network type
    if network_type == 'monopoly_buyer':
        G_seller_buyer = create_monopoly_buyer_network(buyers, sellers)
    elif network_type == 'isolated_buyers':
        G_seller_buyer = create_isolated_buyers_network(buyers, sellers)
    elif network_type == 'monopoly_seller':
        G_seller_buyer = create_monopoly_seller_network(buyers, sellers)
    elif network_type == 'clustered_subgroups':
        G_seller_buyer = create_clustered_subgroups_network(buyers, sellers, num_clusters=num_clusters)
    else:
        G_seller_buyer, G_buyer_buyer = create_seller_shared_buyer_network(buyers, sellers)

    # Create buyer-buyer connections based on shared sellers
    G_buyer_buyer = nx.Graph()
    for buyer in buyers:
        G_buyer_buyer.add_node(buyer.id, obj=buyer)

    for seller in sellers:
        connected_buyers = [buyer for buyer in G_seller_buyer.neighbors(seller.id)]
        for i in range(len(connected_buyers)):
            for j in range(i + 1, len(connected_buyers)):
                G_buyer_buyer.add_edge(connected_buyers[i], connected_buyers[j])

    # Create the seller-seller network based on shared buyers
    G_seller_seller = create_seller_seller_network(buyers, sellers, G_seller_buyer)

    # Simulation loop
    for iteration in range(iterations):
        # Calculate SDR and adjust participants
        SDR = calculate_sdr(buyers, sellers)
        SDR = ou_process_sdr(SDR, mu, theta, sigma, dt)
        adjust_market_participants(SDR, buyers, sellers, SDR_threshold_high, SDR_threshold_low)

        # Ensure price history for any new sellers and buyers
        for seller in sellers:
            if seller.id not in price_history:
                price_history[seller.id] = []
        for buyer in buyers:
            if buyer.id not in buyer_price_history:
                buyer_price_history[buyer.id] = []

        # Apply PSP logic
        update_bids_psp(G_seller_buyer, G_buyer_buyer, G_seller_seller)

        # Track seller and buyer prices over time
        for seller in sellers:
            price_history[seller.id].append(seller.price)
        for buyer in buyers:
            buyer_price_history[buyer.id].append(buyer.bid)

    # Extract seller-buyer connections
    seller_buyer_connections = {}
    for seller in sellers:
        if G_seller_buyer.has_node(seller.id):
            neighbors = list(G_seller_buyer.neighbors(seller.id))
            seller_buyer_connections[seller.id] = [
                buyer for buyer in neighbors 
                if G_seller_buyer.nodes[buyer]['obj'] and 'Buyer' in buyer
            ]

    # Save the final network to GraphML files for reproducibility
    save_network(G_seller_buyer, "seller_buyer_network.graphml")
    save_network(G_seller_seller, "seller_seller_network.graphml")

    # Plot the network and price history
    plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections)

# Example to call the main function with the configuration file
# This should be done in the actual execution environment:
#main_network_simulation_from_file("simulation_config.json")
main_network_simulation_from_file("simulation_config.json", {"seller_buyer": "seller_buyer_network.graphml", "seller_seller": "seller_seller_network.graphml"})