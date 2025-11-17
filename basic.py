import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# Node class for auction participants
class Node:
    def __init__(self, id, price, quantity):
        self.id = id
        self.price = price
        self.quantity = quantity  # Positive for buyers, negative for sellers
        self.last_adjustment = 0  # Buyers remember their last bid adjustment (flash memory)

    def adjust_bid(self, price_adjustment):
        self.last_adjustment = price_adjustment
        self.price += price_adjustment

# Create two small-world networks: One for seller-buyer connections, and one for buyer-buyer connections
def create_small_world_network(num_buyers, num_sellers, k=3, p=0.1):
    G_seller_buyer = nx.Graph()
    G_buyer_buyer = nx.Graph()

    # Add sellers and buyers to the seller-buyer graph
    sellers = [Node(i, price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20)) for i in range(num_sellers)]
    buyers = [Node(i + num_sellers, price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20)) for i in range(num_buyers)]
    
    for seller in sellers:
        G_seller_buyer.add_node(seller.id, obj=seller)
    
    for buyer in buyers:
        G_seller_buyer.add_node(buyer.id, obj=buyer)
        G_buyer_buyer.add_node(buyer.id, obj=buyer)  # Add buyers with same IDs to the buyer-buyer network

    # Connect sellers to buyers in the seller-buyer network
    for buyer in buyers:
        seller = random.choice(sellers)  # Ensure each buyer is connected to at least one seller
        G_seller_buyer.add_edge(seller.id, buyer.id)

    # Create a small-world network among buyers (buyer-buyer connections)
    # Adjust to use the same IDs as in the seller-buyer network
    for edge in nx.watts_strogatz_graph(num_buyers, k, p).edges:
        G_buyer_buyer.add_edge(edge[0] + num_sellers, edge[1] + num_sellers)

    return G_seller_buyer, G_buyer_buyer

# Adjust bids for buyers and sellers based on PSP auction logic
def update_bids_psp(G_seller_buyer, G_buyer_buyer):
    for node_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[node_id]['obj']
        
        # Seller logic: Adjust based on connected buyer bids
        if node.quantity < 0:  # Seller
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(node_id) if G_seller_buyer.nodes[neighbor]['obj'].quantity > 0]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]  # Second highest bid
                price_adjustment = second_highest_bid - node.price
                node.adjust_bid(price_adjustment)
        
        # Buyer logic: Adjust based on connected sellers and buyers
        elif node.quantity > 0:  # Buyer
            seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(node_id) if G_seller_buyer.nodes[neighbor]['obj'].quantity < 0]
            if seller_prices:
                min_seller_price = min(seller_prices)  # Buyers want the lowest seller price
                buyer_bids = [G_buyer_buyer.nodes[neighbor]['obj'].price for neighbor in G_buyer_buyer.neighbors(node_id)]
                if buyer_bids:
                    avg_buyer_influence = np.mean(buyer_bids)  # Average price of connected buyers
                    # Adjust buyer bid: 50% from other buyers, 50% from sellers
                    price_adjustment = (avg_buyer_influence - node.price) * 0.5 + (min_seller_price - node.price) * 0.5
                    node.adjust_bid(price_adjustment)

# Run the auction
def run_auction_psp(num_buyers, num_sellers, num_iterations=50):
    G_seller_buyer, G_buyer_buyer = create_small_world_network(num_buyers, num_sellers)
    
    price_history = {node_id: [G_seller_buyer.nodes[node_id]['obj'].price] for node_id in G_seller_buyer.nodes}
    
    for _ in range(num_iterations):
        update_bids_psp(G_seller_buyer, G_buyer_buyer)
        
        # Record price history
        for node_id in G_seller_buyer.nodes:
            price_history[node_id].append(G_seller_buyer.nodes[node_id]['obj'].price)

    return G_seller_buyer, G_buyer_buyer, price_history
# Visualize the combined network (seller-buyer and buyer-buyer) with clustered colors and limited legend for sellers only
def plot_clustered_network_and_bids_with_sellers_only_legend(G_seller_buyer, G_buyer_buyer, price_history, seller_buyer_connections):
    pos = nx.spring_layout(G_seller_buyer)  # Use the same layout for both networks

    plt.figure(figsize=(12, 10))

    # Basic colors for the 5 sellers in the legend
    basic_colors = ['red', 'blue', 'green', 'purple', 'orange']
    # Full spectrum for actual plot color coding
    full_spectrum_colors = plt.cm.rainbow(np.linspace(0, 1, len(seller_buyer_connections)))  # Generate full spectrum colors for clusters

    # First subplot: Combined seller-buyer and buyer-buyer network with clustered colors
    plt.subplot(2, 1, 1)
    for idx, (seller_id, buyers) in enumerate(seller_buyer_connections.items()):
        color = full_spectrum_colors[idx]  # Use the full spectrum for the actual plot
        # Draw the seller node in the cluster color
        nx.draw_networkx_nodes(G_seller_buyer, pos, nodelist=[seller_id], node_color=[color], node_size=500, label=f"Seller {seller_id}")
        # Draw the buyer nodes in the cluster color
        buyer_ids = [buyer.id for buyer in buyers]
        nx.draw_networkx_nodes(G_seller_buyer, pos, nodelist=buyer_ids, node_color=[color] * len(buyer_ids), node_size=500)
        # Draw seller-buyer edges in the cluster color
        for buyer_id in buyer_ids:
            nx.draw_networkx_edges(G_seller_buyer, pos, edgelist=[(seller_id, buyer_id)], edge_color=[color], style='solid', width=2)

    # Draw the buyer-buyer network (dotted edges)
    dotted_edges = [(u, v) for u, v in G_buyer_buyer.edges]
    nx.draw_networkx_edges(G_buyer_buyer, pos, edgelist=dotted_edges, style='dotted', edge_color='blue', width=1)

    # Add labels and title
    nx.draw_networkx_labels(G_seller_buyer, pos)
    plt.title("Clustered Seller-Buyer and Buyer-Buyer Network")

    # Second subplot: Plot price adjustments over time with clusters and seller-only legend
    plt.subplot(2, 1, 2)

    for idx, (seller_id, buyers) in enumerate(seller_buyer_connections.items()):
        color = full_spectrum_colors[idx]  # Use full spectrum for the plot
        legend_color = basic_colors[idx % len(basic_colors)]  # Use basic colors for legend (limited to 5 sellers)
        # Plot seller's price as a solid line
        plt.plot(np.arange(len(price_history[seller_id])), price_history[seller_id], linestyle='solid', color=color, label=f"Seller {seller_id}")
        # Plot buyers' prices as dotted lines without adding them to the legend
        for buyer in buyers:
            plt.plot(np.arange(len(price_history[buyer.id])), price_history[buyer.id], linestyle='dotted', color=color)

    plt.title("Price Adjustments Over Time (Clustered by Seller)")
    plt.xlabel("Iterations")
    plt.ylabel("Price")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.show()

# Running the PSP auction with clustered price adjustments, colored network, and seller-only legend
def main_psp_with_seller_only_legend():
    num_buyers = 20  # Adjusted to original size
    num_sellers = 5  # Adjusted to original size
    num_iterations = 50

    # Creating seller-buyer and buyer-buyer networks with shared sellers
    G_seller_buyer, G_buyer_buyer = create_seller_shared_buyer_network(num_buyers, num_sellers)
    
    # Run auction and track price history
    price_history = {node_id: [G_seller_buyer.nodes[node_id]['obj'].price] for node_id in G_seller_buyer.nodes}
    for _ in range(num_iterations):
        update_bids_psp(G_seller_buyer, G_buyer_buyer)
        for node_id in G_seller_buyer.nodes:
            price_history[node_id].append(G_seller_buyer.nodes[node_id]['obj'].price)

    # Extract seller-buyer connections
    seller_buyer_connections = {}
    for seller in G_seller_buyer.nodes:
        if G_seller_buyer.nodes[seller]['obj'].quantity < 0:  # Seller
            seller_buyer_connections[seller] = [G_seller_buyer.nodes[neighbor]['obj'] for neighbor in G_seller_buyer.neighbors(seller) if G_seller_buyer.nodes[neighbor]['obj'].quantity > 0]

    # Plot clustered price adjustments and colored network with seller-only legend
    plot_clustered_network_and_bids_with_sellers_only_legend(G_seller_buyer, G_buyer_buyer, price_history, seller_buyer_connections)

# Run the auction with clustered network and price adjustments, with seller-only legend and full spectrum colors
main_psp_with_seller_only_legend()