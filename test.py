import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# Node class for auction participants
class Node:
    def __init__(self, id, price, quantity, mu, theta, sigma):
        self.id = id
        self.price = price
        self.quantity = quantity  # Positive for buyers, negative for sellers
        self.mu = mu  # Long-term mean price for the OU process
        self.theta = theta  # Speed of mean reversion
        self.sigma = sigma  # Volatility
        self.last_adjustment = 0  # Buyers remember their last bid adjustment (flash memory)

    def adjust_bid(self, price_adjustment):
        self.last_adjustment = price_adjustment
        self.price += price_adjustment

    def update_price_ou(self, dt=1):
        """Ornstein-Uhlenbeck process to adjust the price for sellers."""
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
        self.price += self.theta * (self.mu - self.price) * dt + self.sigma * dW

# Create a network where buyers are only connected if they share a seller
def create_seller_shared_buyer_network(num_buyers, num_sellers):
    G_seller_bbuyer = nx.Graph()
    G_bbuyer_bbuyer = nx.Graph()

    # Add sellers and buyers to the seller-buyer graph
    sellers = [Node(i, price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20), mu=np.random.uniform(30, 40), theta=np.random.uniform(0.1, 0.3), sigma=np.random.uniform(1, 5)) for i in range(num_sellers)]
    buyers = [Node(i + num_sellers, price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20), mu=0, theta=0, sigma=0) for i in range(num_buyers)]
    
    for seller in sellers:
        G_seller_bbuyer.add_node(seller.id, obj=seller)
    
    for buyer in buyers:
        G_seller_bbuyer.add_node(buyer.id, obj=buyer)
        G_bbuyer_bbuyer.add_node(buyer.id, obj=buyer)  # Add buyers with same IDs to the buyer-buyer network

    # Connect buyers to sellers in the seller-buyer network
    seller_bbuyer_connections = {}
    for seller in sellers:
        num_connections = min(num_buyers, max(1, random.randint(1, num_buyers)))
        connected_buyers = random.sample(buyers, num_connections)
        seller_bbuyer_connections[seller.id] = connected_buyers
        for buyer in connected_buyers:
            G_seller_bbuyer.add_edge(seller.id, buyer.id)

    # Create the buyer-buyer network by connecting buyers that share the same seller
    for seller_id, buyer_list in seller_bbuyer_connections.items():
        for i in range(len(buyer_list)):
            for j in range(i + 1, len(buyer_list)):
                G_bbuyer_bbuyer.add_edge(buyer_list[i].id, buyer_list[j].id)

    return G_seller_bbuyer, G_bbuyer_bbuyer, seller_bbuyer_connections

# Adjust bids for buyers and sellers based on PSP auction logic
def update_bids_psp(G_seller_bbuyer, G_bbuyer_bbuyer):
    for node_id in G_seller_bbuyer.nodes:
        node = G_seller_bbuyer.nodes[node_id]['obj']
        
        # Seller logic: Adjust based on connected buyer bids
        if node.quantity < 0:  # Seller
            buyer_bids = [G_seller_bbuyer.nodes[neighbor]['obj'].price for neighbor in G_seller_bbuyer.neighbors(node_id) if G_seller_bbuyer.nodes[neighbor]['obj'].quantity > 0]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]  # Second highest bid
                price_adjustment = second_highest_bid - node.price
                node.adjust_bid(price_adjustment)
        
        # Buyer logic: Adjust based on connected sellers and buyers
        elif node.quantity > 0:  # Buyer
            seller_prices = [G_seller_bbuyer.nodes[neighbor]['obj'].price for neighbor in G_seller_bbuyer.neighbors(node_id) if G_seller_bbuyer.nodes[neighbor]['obj'].quantity < 0]
            if seller_prices:
                min_seller_price = min(seller_prices)  # Buyers want the lowest seller price
                buyer_bids = [G_bbuyer_bbuyer.nodes[neighbor]['obj'].price for neighbor in G_bbuyer_bbuyer.neighbors(node_id)]
                if buyer_bids:
                    avg_buyer_influence = np.mean(buyer_bids)  # Average price of connected buyers
                    # Adjust buyer bid: 50% from other buyers, 50% from sellers
                    price_adjustment = (avg_buyer_influence - node.price) * 0.5 + (min_seller_price - node.price) * 0.5
                    node.adjust_bid(price_adjustment)

# Visualize the combined network (seller-buyer and buyer-buyer) with clustered colors and limited legend for sellers only
def plot_clustered_network_and_bids_with_sellers_only_legend(G_seller_bbuyer, G_bbuyer_bbuyer, price_history, seller_bbuyer_connections):
    pos = nx.spring_layout(G_seller_bbuyer)  # Use the same layout for both networks

    plt.figure(figsize=(12, 10))

    # Basic colors for the sellers in the legend
    basic_colors = ['red', 'blue', 'green', 'purple', 'orange']
    # Full spectrum for actual plot color coding
    full_spectrum_colors = plt.cm.rainbow(np.linspace(0, 1, len(seller_bbuyer_connections)))  # Generate full spectrum colors for clusters

    # First subplot: Combined seller-buyer and buyer-buyer network with clustered colors
    plt.subplot(2, 1, 1)
    for idx, (seller_id, buyers) in enumerate(seller_bbuyer_connections.items()):
        color = full_spectrum_colors[idx]  # Use the full spectrum for the actual plot
        # Draw the seller node in the cluster color
        nx.draw_networkx_nodes(G_seller_bbuyer, pos, nodelist=[seller_id], node_color=[color], node_size=500, label=f"Seller {seller_id}")
        # Draw the buyer nodes in the cluster color
        buyer_ids = [buyer.id for buyer in buyers]
        nx.draw_networkx_nodes(G_seller_bbuyer, pos, nodelist=buyer_ids, node_color=[color] * len(buyer_ids), node_size=300)
        # Draw seller-buyer edges in the cluster color
        for buyer_id in buyer_ids:
            nx.draw_networkx_edges(G_seller_bbuyer, pos, edgelist=[(seller_id, buyer_id)], edge_color=[color], style='solid', width=2)

    # Draw the buyer-buyer network (dotted edges)
    dotted_edges = [(u, v) for u, v in G_bbuyer_bbuyer.edges]
    nx.draw_networkx_edges(G_bbuyer_bbuyer, pos, edgelist=dotted_edges, style='dotted', edge_color='gray', width=1)

    # Add labels and title
    nx.draw_networkx_labels(G_seller_bbuyer, pos)
    plt.title("Clustered Seller-Buyer and Buyer-Buyer Network")

    # Second subplot: Plot price adjustments over time with clusters and seller-only legend
    plt.subplot(2, 1, 2)

    for idx, (seller_id, buyers) in enumerate(seller_bbuyer_connections.items()):
        color = full_spectrum_colors[idx]  # Use full spectrum for the plot
        legend_color = basic_colors[idx % len(basic_colors)]  # Use basic colors for legend (limited to number of colors)
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

# Main simulation function with seller-specific OU processes
def main_psp_with_seller_ou_processes(iterations=100):
    num_buyers = 10
    num_sellers = 3

    # Initialize buyers and sellers with quantities and unique OU parameters for sellers
    buyers = [Node(i, price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20), mu=0, theta=0, sigma=0) for i in range(num_buyers)]
    sellers = [
        Node(i + num_buyers, price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20), 
             mu=np.random.uniform(30, 40), theta=np.random.uniform(0.1, 0.3), sigma=np.random.uniform(1, 5))
        for i in range(num_sellers)
    ]

    # Run auction and track price history
    price_history = {node.id: [node.price] for node in buyers + sellers}

    for iteration in range(iterations):
        # Update each seller's price using its own OU process
        for seller in sellers:
            seller.update_price_ou(dt=1)

        # Create new networks with adjusted buyers and sellers
        G_seller_bbuyer, G_bbuyer_bbuyer, seller_bbuyer_connections = create_seller_shared_buyer_network(len(buyers), len(sellers))

        # Update bids and track price history
        update_bids_psp(G_seller_bbuyer, G_bbuyer_bbuyer)
        for node_id in G_seller_bbuyer.nodes:
            node = G_seller_bbuyer.nodes[node_id]['obj']
            if node_id not in price_history:
                price_history[node_id] = [node.price]
            else:
                price_history[node_id].append(node.price)

    # Plot clustered price adjustments and colored network with seller-only legend
    plot_clustered_network_and_bids_with_sellers_only_legend(G_seller_bbuyer, G_bbuyer_bbuyer, price_history, seller_bbuyer_connections)

# Run the auction with seller-specific OU processes
main_psp_with_seller_ou_processes(iterations=1000)