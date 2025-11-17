import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import json

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
        self.quantity = quantity  # Positive for buyers, negative for sellers
        self.bid = np.random.uniform(50, 100)  # Initial bid for buyers

# Create a network where buyers are connected to multiple sellers
def create_seller_shared_buyer_network(buyers, sellers):
    print("Creating shared network")
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
    
# Create seller-seller network based on shared buyers
def create_seller_seller_network(buyers, sellers, G_seller_buyer):
    print("Creating seller-seller network based on shared buyers")
    
    # Initialize an empty graph for seller-seller relationships
    G_seller_seller = nx.Graph()

    # Add all sellers as nodes in the seller-seller network
    for seller in sellers:
        G_seller_seller.add_node(seller.id, obj=seller)
    
    # Iterate through all pairs of sellers to find shared buyers
    for i, seller_1 in enumerate(sellers):
        for seller_2 in sellers[i+1:]:
            # Find common buyers between seller_1 and seller_2
            buyers_seller_1 = set(G_seller_buyer.neighbors(seller_1.id))
            buyers_seller_2 = set(G_seller_buyer.neighbors(seller_2.id))
            shared_buyers = buyers_seller_1.intersection(buyers_seller_2)
            
            if shared_buyers:
                # Add an edge between seller_1 and seller_2 if they share buyers
                G_seller_seller.add_edge(seller_1.id, seller_2.id, shared_buyers=list(shared_buyers))

    return G_seller_seller
    
# Create a monopoly buyer network
def create_monopoly_buyer_network(buyers, sellers):
    print("Creating monopoly buyer network")
    G_seller_buyer = nx.Graph()
    
    # One buyer (the "monopoly" buyer) connected to all sellers
    monopoly_buyer = buyers[0]
    G_seller_buyer.add_node(monopoly_buyer.id, obj=monopoly_buyer)
    for seller in sellers:
        G_seller_buyer.add_node(seller.id, obj=seller)
        G_seller_buyer.add_edge(seller.id, monopoly_buyer.id)

    # Other buyers are only connected to one seller each
    for i, buyer in enumerate(buyers[1:]):
        G_seller_buyer.add_node(buyer.id, obj=buyer)
        G_seller_buyer.add_edge(sellers[i % len(sellers)].id, buyer.id)

    return G_seller_buyer

# Create isolated buyers network
def create_isolated_buyers_network(buyers, sellers):
    print("Creating isolated buyer network")
    G_seller_buyer = nx.Graph()

    for i, buyer in enumerate(buyers):
        seller = sellers[i % len(sellers)]
        G_seller_buyer.add_node(buyer.id, obj=buyer)
        G_seller_buyer.add_node(seller.id, obj=seller)
        G_seller_buyer.add_edge(seller.id, buyer.id)

    return G_seller_buyer

# Create monopoly seller network
def create_monopoly_seller_network(buyers, sellers):
    print("Creating monopoly seller network")
    G_seller_buyer = nx.Graph()

    monopoly_seller = sellers[0]
    G_seller_buyer.add_node(monopoly_seller.id, obj=monopoly_seller)
    for buyer in buyers:
        G_seller_buyer.add_node(buyer.id, obj=buyer)
        G_seller_buyer.add_edge(monopoly_seller.id, buyer.id)

    # Other sellers are only connected to one buyer each
    for i, seller in enumerate(sellers[1:]):
        G_seller_buyer.add_node(seller.id, obj=seller)
        G_seller_buyer.add_edge(seller.id, buyers[i % len(buyers)].id)

    return G_seller_buyer

# Create clustered subgroups network
def create_clustered_subgroups_network(buyers, sellers, num_clusters=2):
    print("Creating clustered subgroups network")
    G_seller_buyer = nx.Graph()
    
    # Split buyers and sellers into clusters
    clusters = []
    for i in range(num_clusters):
        buyer_cluster = buyers[i::num_clusters]
        seller_cluster = sellers[i::num_clusters]
        clusters.append((buyer_cluster, seller_cluster))

    # Create edges within each cluster
    for buyer_cluster, seller_cluster in clusters:
        for buyer in buyer_cluster:
            for seller in seller_cluster:
                G_seller_buyer.add_node(buyer.id, obj=buyer)
                G_seller_buyer.add_node(seller.id, obj=seller)
                G_seller_buyer.add_edge(seller.id, buyer.id)

    return G_seller_buyer


# Update the price logic to include influence from the seller-seller network
def update_bids_psp(G_seller_buyer, G_buyer_buyer, G_seller_seller):
    for seller_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[seller_id]['obj']
        
        # Seller logic: Adjust based on connected buyer bids and seller-seller influence
        if "Seller" in seller_id:  # Seller
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(seller_id)]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]  # Progressive Second Price: Second-highest bid
                node.price = second_highest_bid  # Sellers price at second-highest buyer bid

            # Seller-Seller influence: Adjust based on prices of connected sellers
            connected_seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_seller.neighbors(seller_id)]
            if connected_seller_prices:
                avg_seller_price = np.mean(connected_seller_prices)
                # Mix current price with the average price of connected sellers
                node.price = (node.price + avg_seller_price) / 2

        # Buyer logic: Adjust based on connected sellers and other buyers (PSP influenced)
        elif "Buyer" in seller_id:  # Buyer
            seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(seller_id) if "Seller" in neighbor]
            if seller_prices:
                min_seller_price = min(seller_prices)  # Buyers prefer the lowest price
                # Adjust bid based on both seller prices and competition from other buyers
                other_buyer_bids = [G_buyer_buyer.nodes[neighbor]['obj'].bid for neighbor in G_buyer_buyer.neighbors(seller_id)]
                if other_buyer_bids:
                    second_highest_buyer_bid = sorted(other_buyer_bids)[-2] if len(other_buyer_bids) > 1 else other_buyer_bids[0]
                    G_seller_buyer.nodes[seller_id]['obj'].bid = (min_seller_price + second_highest_buyer_bid) / 2  # Adjust bid competitively


# Plot the network and price history
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

# Calculate Supply-Demand Ratio (SDR)
def calculate_sdr(buyers, sellers):
    total_supply = abs(sum(seller.quantity for seller in sellers))
    total_demand = sum(buyer.quantity for buyer in buyers)
    if total_demand == 0:
        return float('inf')
    return total_supply / total_demand

# OU Process for adjusting SDR
def ou_process_sdr(SDR, mu, theta, sigma, dt):
    dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
    SDR += theta * (mu - SDR) * dt + sigma * dW
    return SDR

# Adjust market participants based on SDR
def adjust_market_participants(SDR, buyers, sellers, SDR_threshold_high, SDR_threshold_low):
    if SDR > SDR_threshold_high:  # Too much supply
        if len(sellers) > 0:
            removed_seller = sellers.pop()  # Remove a random seller
            print(f"Removed Seller: {removed_seller.id}")
        if random.random() < 0.5:
            new_buyer = Node(f"Buyer_{len(buyers)}", price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20))
            buyers.append(new_buyer)
            print(f"Added Buyer: {new_buyer.id}")

    elif SDR < SDR_threshold_low:  # Too much demand
        if len(buyers) > 0:
            removed_buyer = buyers.pop()  # Remove a random buyer
            print(f"Removed Buyer: {removed_buyer.id}")
        if random.random() < 0.5:
            new_seller = Node(f"Seller_{len(sellers)}", price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20))
            sellers.append(new_seller)
            print(f"Added Seller: {new_seller.id}")
    
    
# Main function to load parameters from a file and save networks after simulation
def main_network_simulation(config_file):
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


    # Initialize buyers and sellers
    buyers = [Node(f"Buyer_{i}", price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20)) for i in range(num_buyers)]
    sellers = [Node(f"Seller_{i}", price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20)) for i in range(num_sellers)]

    price_history = {seller.id: [] for seller in sellers}
    buyer_price_history = {buyer.id: [] for buyer in buyers}

    dt = 1

    SDR_threshold_high = 1.5  # Too much supply
    SDR_threshold_low = 0.7  # Too much demand

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

    # Connect buyers through shared sellers
    for seller in sellers:
        connected_buyers = [buyer for buyer in G_seller_buyer.neighbors(seller.id)]
        for i in range(len(connected_buyers)):
            for j in range(i + 1, len(connected_buyers)):
                G_buyer_buyer.add_edge(connected_buyers[i], connected_buyers[j])
                
    # Create the seller-seller network based on shared buyers
    G_seller_seller = create_seller_seller_network(buyers, sellers, G_seller_buyer)

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
        # Check if the seller exists in the graph before trying to access its neighbors
        if G_seller_buyer.has_node(seller.id):
            print(f"Seller {seller.id} found in G_seller_buyer.")
            
            # Retrieve the neighbors of the seller
            neighbors = list(G_seller_buyer.neighbors(seller.id))
            print(f"Seller {seller.id} neighbors: {neighbors}")
            
            # Check that each neighbor is labeled as a buyer (i.e., 'Buyer' is in the ID)
            seller_buyer_connections[seller.id] = [
                buyer for buyer in neighbors 
                if G_seller_buyer.nodes[buyer]['obj'] and 'Buyer' in buyer
            ]
        else:
            # If seller is not found in the graph, print an error message
            print(f"Seller {seller.id} not found in G_seller_buyer.")
            
    # Plot network and price history
    plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections)


main_network_simulation("SAF:Storage/Code/simulation_config.json")
#main_network_simulation("simulation_config.json", {"seller_buyer": "seller_buyer_network.graphml", "seller_seller": "seller_seller_network.graphml"})