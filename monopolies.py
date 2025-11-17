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

# Create a monopoly buyer network
def create_monopoly_buyer_network(buyers, sellers):
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
    G_seller_buyer = nx.Graph()

    for i, buyer in enumerate(buyers):
        seller = sellers[i % len(sellers)]
        G_seller_buyer.add_node(buyer.id, obj=buyer)
        G_seller_buyer.add_node(seller.id, obj=seller)
        G_seller_buyer.add_edge(seller.id, buyer.id)

    return G_seller_buyer

# Create monopoly seller network
def create_monopoly_seller_network(buyers, sellers):
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

# Main function for running different network structures
def main_network_simulation(network_type='monopoly_buyer', iterations=100):
    num_buyers = 5
    num_sellers = 3

    # Initialize buyers and sellers
    buyers = [Node(f"Buyer_{i}", price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20)) for i in range(num_buyers)]
    sellers = [Node(f"Seller_{i}", price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20)) for i in range(num_sellers)]

    price_history = {seller.id: [] for seller in sellers}
    buyer_price_history = {buyer.id: [] for buyer in buyers}

    # Create the selected network type
    if network_type == 'monopoly_buyer':
        G_seller_buyer = create_monopoly_buyer_network(buyers, sellers)
    elif network_type == 'isolated_buyers':
        G_seller_buyer = create_isolated_buyers_network(buyers, sellers)
    elif network_type == 'monopoly_seller':
        G_seller_buyer = create_monopoly_seller_network(buyers, sellers)
    elif network_type == 'clustered_subgroups':
        G_seller_buyer = create_clustered_subgroups_network(buyers, sellers, num_clusters=2)

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

    for iteration in range(iterations):
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

    # Plot network and price history with original color scheme and shared buyer connections
    plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections)

# Run the simulation for different network types
main_network_simulation('monopoly_buyer', iterations=50)

# To run isolated buyers, use: main_network_simulation('isolated_buyers', iterations=50)
# To run monopoly seller, use: main_network_simulation('monopoly_seller', iterations=50)
# To run clustered subgroups, use: main_network_simulation('clustered_subgroups', iterations=50)