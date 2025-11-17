import numpy as np
import random
import networkx as nx
from collections import deque

# Opt-out function that checks if a node should opt out based on influence sets
def opt_out_function(node_id, G_seller_buyer, G_buyer_buyer, G_seller_seller, influence_threshold, window_size=5):
    """
    Determines if a node (buyer or seller) should opt out of the market.
    
    :param node_id: ID of the node (buyer or seller)
    :param G_seller_buyer: Buyer-Seller network
    :param G_buyer_buyer: Buyer-Buyer network
    :param G_seller_seller: Seller-Seller network
    :param influence_threshold: Threshold below which the node opts out
    :param window_size: Time window size for calculating moving average influence
    :return: True if node opts out, False otherwise
    """
    # Get the node object
    node = G_seller_buyer.nodes[node_id]['obj']
    
    # Initialize influence sets
    influence_set = deque(maxlen=window_size)
    
    # Calculate influence from buyer-seller connections
    if "Buyer" in node_id:
        # Buyers are influenced by sellers
        seller_influence = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(node_id) if "Seller" in neighbor]
        if seller_influence:
            avg_seller_influence = np.mean(seller_influence)
            influence_set.append(avg_seller_influence)
    
    elif "Seller" in node_id:
        # Sellers are influenced by buyers and other sellers
        buyer_influence = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(node_id) if "Buyer" in neighbor]
        seller_influence = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_seller.neighbors(node_id)]
        
        if buyer_influence:
            avg_buyer_influence = np.mean(buyer_influence)
            influence_set.append(avg_buyer_influence)
        if seller_influence:
            avg_seller_influence = np.mean(seller_influence)
            influence_set.append(avg_seller_influence)
    
    # Calculate the moving average of the influence set
    if len(influence_set) == window_size:
        avg_influence = np.mean(influence_set)
        # Opt-out if the influence is below the threshold
        if avg_influence < influence_threshold:
            print(f"Node {node_id} opts out due to low influence: {avg_influence}")
            return True
    
    return False

# Update bids and handle opt-out
def update_bids_psp_with_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config, influence_threshold):
    """
    Update bids using the PSP model while considering the opt-out mechanism based on influence sets.
    
    :param G_seller_buyer: Buyer-Seller network
    :param G_buyer_buyer: Buyer-Buyer network
    :param G_seller_seller: Seller-Seller network
    :param buyer_config: Configuration for buyers
    :param seller_config: Configuration for sellers
    :param influence_threshold: Threshold for opt-out decision
    """
    for node_id in G_seller_buyer.nodes:
        # Check if the node opts out based on influence sets
        if opt_out_function(node_id, G_seller_buyer, G_buyer_buyer, G_seller_seller, influence_threshold):
            continue  # Skip if the node opts out
        
        node = G_seller_buyer.nodes[node_id]['obj']
        
        # Seller logic: Adjust price based on bids from connected buyers
        if "Seller" in node_id:
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(node_id) if "Buyer" in neighbor]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]  # Second highest bid
                node.price = second_highest_bid
        
        # Buyer logic: Adjust bid based on seller prices
        elif "Buyer" in node_id:
            seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(node_id) if "Seller" in neighbor]
            if seller_prices:
                min_seller_price = min(seller_prices)
                other_buyer_bids = [G_buyer_buyer.nodes[neighbor]['obj'].bid for neighbor in G_buyer_buyer.neighbors(node_id)]
                if other_buyer_bids:
                    second_highest_buyer_bid = sorted(other_buyer_bids)[-2] if len(other_buyer_bids) > 1 else other_buyer_bids[0]
                    node.bid = (min_seller_price + second_highest_buyer_bid) / 2

# Incorporate this into the main simulation loop
def main_network_simulation_with_opt_out():
    # Your existing simulation setup...
    # For example, setting up the networks and initializing buyers and sellers
    
    config = {
        "network_type": "clustered_subgroups",
        "num_buyers": 20,
        "num_sellers": 5,
        "iterations": 20,
        "num_clusters": 5,
        "influence_threshold": 50.0,  # Example threshold for opting out
    }

    # Initialize your networks (buyer-seller, buyer-buyer, seller-seller)
    # This can be from your existing simulation setup
    
    # Run the simulation loop
    for iteration in range(config['iterations']):
        print(f"Iteration {iteration}")
        
        # Update bids with opt-out mechanism
        update_bids_psp_with_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config, config['influence_threshold'])
        
        # You can also integrate the opt-out decisions into the price history tracking and visualizations
        # Track price history, adjust network as needed

    # End of the simulation, plot results as needed