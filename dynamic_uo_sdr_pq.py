import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# Node class for auction participants (includes proportional allocation logic)
class Node:
    def __init__(self, id, price, quantity):
        self.id = id
        self.price = price
        self.quantity = quantity  # Positive for buyers, negative for sellers
        self.last_adjustment = 0  # Buyers remember their last bid adjustment (flash memory)
        self.allocated_quantity = 0  # Track allocated quantity for buyers

    def adjust_bid(self, price_adjustment):
        self.last_adjustment = price_adjustment
        self.price += price_adjustment

    def allocate_quantity(self, total_quantity, proportion):
        """Allocate quantity to buyer based on proportion of their bid to total bids."""
        self.allocated_quantity = total_quantity * proportion

# Mean reverting process for SDR
def mean_reverting_sdr(SDR, mu, theta, sigma, dt):
    """Ornstein-Uhlenbeck process to model the supply-demand ratio (SDR)."""
    dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
    SDR += theta * (mu - SDR) * dt + sigma * dW
    return SDR

# Calculate SDR based on quantities of buyers and sellers
def calculate_sdr(buyers, sellers):
    """Calculate SDR as the ratio of total supply (seller quantity) to total demand (buyer quantity)."""
    total_supply = abs(sum(seller.quantity for seller in sellers))  # Sum of negative quantities from sellers
    total_demand = sum(buyer.quantity for buyer in buyers)  # Sum of positive quantities from buyers

    if total_demand == 0:
        return float('inf')  # Avoid division by zero, treat infinite SDR as extreme oversupply
    else:
        return total_supply / total_demand

# Adjust buyers and sellers based on the SDR calculated from total quantities
def adjust_market_participants_based_on_quantity(SDR, buyers, sellers, SDR_threshold_high, SDR_threshold_low):
    """Adjust buyers and sellers entering/leaving based on SDR calculated from quantities."""
    if SDR > SDR_threshold_high:
        # Too much supply: sellers leave, buyers enter
        if len(sellers) > 0:
            sellers.pop()  # Randomly remove one seller
        if random.random() < 0.5:  # 50% chance for a buyer to enter
            buyers.append(Node(len(buyers) + len(sellers), price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20)))
    elif SDR < SDR_threshold_low:
        # Too much demand: buyers leave, sellers enter
        if len(buyers) > 0:
            buyers.pop()  # Randomly remove one buyer
        if random.random() < 0.5:  # 50% chance for a seller to enter
            sellers.append(Node(len(buyers) + len(sellers), price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20)))
    return buyers, sellers

# Create a network where buyers are only connected if they share a seller
def create_seller_shared_buyer_network(num_buyers, num_sellers):
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

    # Connect buyers to sellers in the seller-buyer network
    seller_buyer_connections = {}
    for seller in sellers:
        # Ensure there is always at least one buyer for every seller
        num_connections = min(num_buyers, max(1, random.randint(1, num_buyers)))
        connected_buyers = random.sample(buyers, num_connections)
        seller_buyer_connections[seller.id] = connected_buyers
        for buyer in connected_buyers:
            G_seller_buyer.add_edge(seller.id, buyer.id)

    # Create the buyer-buyer network by connecting buyers that share the same seller
    for seller_id, buyer_list in seller_buyer_connections.items():
        for i in range(len(buyer_list)):
            for j in range(i + 1, len(buyer_list)):
                G_buyer_buyer.add_edge(buyer_list[i].id, buyer_list[j].id)

    return G_seller_buyer, G_buyer_buyer

# Update bids and proportional allocation based on PSP logic
def update_bids_psp_with_proportional_allocation(G_seller_buyer, G_buyer_buyer):
    for node_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[node_id]['obj']
        
        # Seller logic: Adjust based on connected buyer bids and allocate quantity proportionally
        if node.quantity < 0:  # Seller
            buyer_bids = [(G_seller_buyer.nodes[neighbor]['obj'].price, neighbor) for neighbor in G_seller_buyer.neighbors(node_id) if G_seller_buyer.nodes[neighbor]['obj'].quantity > 0]
            
            if buyer_bids:
                total_bid = sum(bid for bid, _ in buyer_bids)
                
                # Allocate quantity proportionally to buyers
                for bid, buyer_id in buyer_bids:
                    buyer_node = G_seller_buyer.nodes[buyer_id]['obj']
                    proportion = bid / total_bid
                    buyer_node.allocate_quantity(-node.quantity, proportion)

                # Seller adjusts price based on the second-highest bid (PSP logic)
                if len(buyer_bids) > 1:
                    second_highest_bid = sorted(bid for bid, _ in buyer_bids)[-2]  # Second highest bid
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

# Plot clustered price adjustments and colored network with proportional allocation
def plot_clustered_network_and_bids_with_allocation(G_seller_buyer, G_buyer_buyer, price_history, seller_buyer_connections):
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

# Main function for running the auction with quantity-based SDR logic
def main_psp_with_dynamic_market_and_quantity_sdr(iterations=1000):
    num_buyers = 20
    num_sellers = 5

    # Initialize buyers and sellers
    buyers = [Node(i, price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20)) for i in range(num_buyers)]
    sellers = [Node(i + num_buyers, price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20)) for i in range(num_sellers)]

    # Run auction and track price history
    price_history = {node.id: [node.price] for node in buyers + sellers}

    for iteration in range(iterations):
        # Calculate SDR based on total quantity of buyers and sellers
        SDR = calculate_sdr(buyers, sellers)

        # Adjust SDR using mean-reverting process
        # sigma = 0.05
        SDR = mean_reverting_sdr(SDR, mu=1, theta=0.1, sigma=0, dt=1)

        # Adjust buyers and sellers entering/leaving the market based on the new SDR (quantity-based)
        SDR_threshold_high = 2 #1.5  # Threshold for too much supply
        SDR_threshold_low = 0 #0.7   # Threshold for too much demand
        buyers, sellers = adjust_market_participants_based_on_quantity(SDR, buyers, sellers, SDR_threshold_high, SDR_threshold_low)

        # Create new networks with adjusted buyers and sellers
        G_seller_buyer, G_buyer_buyer = create_seller_shared_buyer_network(len(buyers), len(sellers))
        
        # Update bids and track price history with proportional allocation
        update_bids_psp_with_proportional_allocation(G_seller_buyer, G_buyer_buyer)
        for node_id in G_seller_buyer.nodes:
            price_history[node_id].append(G_seller_buyer.nodes[node_id]['obj'].price)

    # Extract seller-buyer connections for the final state
    seller_buyer_connections = {}
    for seller in G_seller_buyer.nodes:
        if G_seller_buyer.nodes[seller]['obj'].quantity < 0:  # Seller
            seller_buyer_connections[seller] = [G_seller_buyer.nodes[neighbor]['obj'] for neighbor in G_seller_buyer.neighbors(seller) if G_seller_buyer.nodes[neighbor]['obj'].quantity > 0]

    # Plot clustered price adjustments and colored network with proportional allocation
    plot_clustered_network_and_bids_with_allocation(G_seller_buyer, G_buyer_buyer, price_history, seller_buyer_connections)

# Run the auction with dynamic entry/exit of buyers and sellers, and SDR based on total quantity
main_psp_with_dynamic_market_and_quantity_sdr(iterations=100)  # Testing with fewer iterations