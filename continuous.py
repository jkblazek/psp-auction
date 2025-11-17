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

# Small-world auction network creation with enforced buyer-seller connections
def create_auction_network_enforced(num_buyers, num_sellers, k=3, p=0.1):
    G = nx.Graph()
    
    # Add sellers (negative quantity) and buyers (positive quantity)
    sellers = [Node(i, price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20)) for i in range(num_sellers)]
    buyers = [Node(i + num_sellers, price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20)) for i in range(num_buyers)]
    
    for seller in sellers:
        G.add_node(seller.id, obj=seller)

    for buyer in buyers:
        G.add_node(buyer.id, obj=buyer)

    # Connect each buyer to at least one seller
    for buyer in buyers:
        seller = np.random.choice(sellers)
        G.add_edge(seller.id, buyer.id, connection_type='seller-buyer')

    # Connect additional buyers to sellers and establish buyer-buyer influence
    for seller in sellers:
        connected_buyers = np.random.choice(buyers, k, replace=False)
        for buyer in connected_buyers:
            G.add_edge(seller.id, buyer.id, connection_type='seller-buyer')
            # Connect buyers to each other who are connected to the same seller
            for other_buyer in connected_buyers:
                if buyer.id != other_buyer.id:
                    G.add_edge(buyer.id, other_buyer.id, connection_type='buyer-buyer')

    return G

# Mean-reverting process for supply-demand balance
def mean_reverting_adjustment(current_price, target_price, alpha=0.05):
    return current_price + alpha * (target_price - current_price)

# Example target price calculation (based on initial prices)
def calculate_average_initial_price(G):
    initial_prices = [G.nodes[node]['obj'].price for node in G.nodes]
    return np.mean(initial_prices)

# Logic to remove satisfied nodes and replace them with new nodes
def check_and_replace_nodes(G, buyers, sellers):
    nodes_to_remove = []
    
    # Identify buyers and sellers that have satisfied their demand or supply
    for node_id in G.nodes:
        node = G.nodes[node_id]['obj']
        if node.quantity <= 0 and node.quantity > -1e-5:  # Buyer demand is satisfied
            nodes_to_remove.append(node_id)
            new_buyer = Node(node_id, price=np.random.uniform(50, 100), quantity=np.random.uniform(10, 20))
            buyers.append(new_buyer)
            G.add_node(new_buyer.id, obj=new_buyer)
            connect_buyer_to_seller(G, new_buyer, sellers)
        elif node.quantity >= 0 and node.quantity < 1e-5:  # Seller supply is satisfied
            nodes_to_remove.append(node_id)
            new_seller = Node(node_id, price=np.random.uniform(20, 50), quantity=-np.random.uniform(10, 20))
            sellers.append(new_seller)
            G.add_node(new_seller.id, obj=new_seller)
            connect_seller_to_buyers(G, new_seller, buyers)

    # Remove satisfied nodes from the market
    for node_id in nodes_to_remove:
        G.remove_node(node_id)

# Helper function to connect a new buyer to at least one seller
def connect_buyer_to_seller(G, buyer, sellers):
    seller = random.choice(sellers)
    G.add_edge(seller.id, buyer.id, connection_type='seller-buyer')

# Helper function to connect a new seller to some buyers
def connect_seller_to_buyers(G, seller, buyers, k=3):
    connected_buyers = random.sample(buyers, min(k, len(buyers)))
    for buyer in connected_buyers:
        G.add_edge(seller.id, buyer.id, connection_type='seller-buyer')
        for other_buyer in connected_buyers:
            if buyer.id != other_buyer.id:
                G.add_edge(buyer.id, other_buyer.id, connection_type='buyer-buyer')

# Adjust bids using PSP logic for both buyers and sellers
def update_bids_psp_with_mean_reversion(G, target_price):
    for node_id in G.nodes:
        node = G.nodes[node_id]['obj']
        
        # Seller logic: Adjust based on buyers' second-highest bid
        if node.quantity < 0:  # Seller
            buyer_bids = [G.nodes[neighbor]['obj'].price for neighbor in G.neighbors(node_id) if G.nodes[neighbor]['obj'].quantity > 0]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]  # Second highest bid
                price_adjustment = second_highest_bid - node.price
                # Add mean-reverting adjustment
                node.adjust_bid(mean_reverting_adjustment(node.price, target_price) + price_adjustment)
        
        # Buyer logic: Adjust based on sellers' prices and second-highest buyer influence
        elif node.quantity > 0:  # Buyer
            seller_prices = [G.nodes[neighbor]['obj'].price for neighbor in G.neighbors(node_id) if G.nodes[neighbor]['obj'].quantity < 0]
            if seller_prices:
                min_seller_price = min(seller_prices)  # Buyers are interested in the lowest seller price
                buyer_bids = [G.nodes[neighbor]['obj'].price for neighbor in G.neighbors(node_id) if G.nodes[neighbor]['obj'].quantity > 0]
                if len(buyer_bids) > 1:
                    second_highest_bid = sorted(buyer_bids)[-2]  # Second highest bid from other buyers
                    price_adjustment = (second_highest_bid - min_seller_price) * 0.1  # Adjust by 10% of difference
                    # Add mean-reverting adjustment
                    node.adjust_bid(mean_reverting_adjustment(node.price, target_price) + price_adjustment)

# Run the auction with node replacement logic
def run_auction_with_replacement(num_iterations, G, buyers, sellers):
    price_history = {node_id: [G.nodes[node_id]['obj'].price] for node_id in G.nodes}
    
    for _ in range(num_iterations):
        target_price = calculate_average_initial_price(G)
        update_bids_psp_with_mean_reversion(G, target_price)
        check_and_replace_nodes(G, buyers, sellers)
        for node_id in G.nodes:
            price_history[node_id].append(G.nodes[node_id]['obj'].price)

    return price_history

# Visualize the auction results with separate graphs and labels
def plot_auction_results_separate_with_labels(G, price_history):
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(12, 8))

    # First subplot: Network visualization
    plt.subplot(1, 2, 1)
    solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d['connection_type'] == 'seller-buyer']
    dotted_edges = [(u, v) for u, v, d in G.edges(data=True) if d['connection_type'] == 'buyer-buyer']
    
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if G.nodes[n]['obj'].quantity < 0], node_color='red', label='Sellers', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if G.nodes[n]['obj'].quantity > 0], node_color='green', label='Buyers', node_size=500)
    
    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, style='solid', edge_color='black', width=2, label='Buyer-Seller Connection')
    nx.draw_networkx_edges(G, pos, edgelist=dotted_edges, style='dotted', edge_color='blue', width=1, label='Buyer-Buyer Connection')
    
    nx.draw_networkx_labels(G, pos)

    plt.title('Auction Network')
    plt.legend(loc='upper left')

    # Second subplot: Price history
    plt.subplot(1, 2, 2)
    for node_id, prices in price_history.items():
        x = np.linspace(0, len(prices)-1, len(prices))
        if G.nodes[node_id]['obj'].quantity > 0:
            plt.plot(x, prices, linestyle='dotted', label=f"Buyer {node_id} price")
        else:
            plt.plot(x, prices, linestyle='solid', label=f"Seller {node_id} price")
    
    plt.title('Bid Price Adjustments Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main function to run the PSP auction with replacement logic
def main_psp_with_replacement():
    num_buyers = 10
    num_sellers = 3
    num_iterations = 50
    
    G = create_auction_network_enforced(num_buyers, num_sellers)
    
    # Initialize buyer and seller lists
    buyers = [G.nodes[node]['obj'] for node in G.nodes if G.nodes[node]['obj'].quantity > 0]
    sellers = [G.nodes[node]['obj'] for node in G.nodes if G.nodes[node]['obj'].quantity < 0]
    
    # Run auction process with node replacement
    price_history = run_auction_with_replacement(num_iterations, G, buyers, sellers)
    
    # Visualize auction results
    plot_auction_results_separate_with_labels(G, price_history)

# Running the PSP auction with replacement logic
main_psp_with_replacement()