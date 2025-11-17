import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from mpl_toolkits.mplot3d import Axes3D


# Modified to track convergence
class Node:
    def __init__(self, id, price, quantity, gamma=1.0):
        self.id = id
        self.price = price
        self.quantity = quantity  # Positive for buyers, negative for sellers
        self.bid = np.random.uniform(50, 100)  # Initial bid for buyers
        self.gamma = gamma  # Elasticity parameter for buyers
        self.valuation_function = lambda z: self.gamma * np.log(1 + z)

# Define a convergence threshold
CONVERGENCE_THRESHOLD = 1e-3
MAX_ITERATIONS = 1000  # Set a maximum iteration limit to prevent infinite loops

# Function to calculate the Euclidean norm of the change in bids or prices
def calculate_convergence(bids_or_prices_previous, bids_or_prices_current):
    common_keys = set(bids_or_prices_previous.keys()).intersection(bids_or_prices_current.keys())
    previous_bids_list = [bids_or_prices_previous[node_id] for node_id in sorted(common_keys)]
    current_bids_list = [bids_or_prices_current[node_id] for node_id in sorted(common_keys)]
    return np.linalg.norm(np.array(previous_bids_list) - np.array(current_bids_list))

# Function to track convergence
def has_converged(previous_bids, current_bids):
    return calculate_convergence(previous_bids, current_bids) < CONVERGENCE_THRESHOLD

# Create a random market network and the buyer-buyer network
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

# Create buyer-buyer network based on shared sellers
def create_buyer_buyer_network(buyers, sellers, G_seller_buyer):
    G_buyer_buyer = nx.Graph()
    for buyer in buyers:
        G_buyer_buyer.add_node(buyer.id, obj=buyer)

    for seller in sellers:
        connected_buyers = [neighbor for neighbor in G_seller_buyer.neighbors(seller.id)]
        for i in range(len(connected_buyers)):
            for j in range(i + 1, len(connected_buyers)):
                G_buyer_buyer.add_edge(connected_buyers[i], connected_buyers[j])

    return G_buyer_buyer

# Opt-out function based on utility maximization (simplified for this example)
def opt_out_utility_based(buyer, connected_sellers, G_seller_buyer, gamma=1.0):
    total_demand = buyer.quantity
    chosen_sellers = random.sample(connected_sellers, k=random.randint(1, len(connected_sellers)))
    return chosen_sellers

# Modify the update bids function to track convergence
def update_bids_with_utility_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config, use_opt_out=True):
    previous_bids = {node_id: G_seller_buyer.nodes[node_id]['obj'].bid for node_id in G_seller_buyer.nodes if "Buyer" in node_id}
    iteration = 0
    converged = False
    
    while not converged and iteration < MAX_ITERATIONS:
        iteration += 1
        current_bids = {}
        
        for node_id in G_seller_buyer.nodes:
            node = G_seller_buyer.nodes[node_id]['obj']

            if "Buyer" in node_id:
                connected_sellers = [G_seller_buyer.nodes[neighbor]['obj'] for neighbor in G_seller_buyer.neighbors(node_id) if "Seller" in neighbor]
                if use_opt_out and len(connected_sellers) > 2:
                    chosen_sellers = opt_out_utility_based(node, connected_sellers, G_seller_buyer)
                    for seller in chosen_sellers:
                        G_seller_buyer.add_edge(node_id, seller.id)
                        node_bid = node.bid
                        seller_price = seller.price
                        node.bid = (node_bid + seller_price) / 2
                else:
                    seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(node_id) if "Seller" in neighbor]
                    if seller_prices:
                        min_seller_price = min(seller_prices)
                        other_buyer_bids = [G_buyer_buyer.nodes[neighbor]['obj'].bid for neighbor in G_buyer_buyer.neighbors(node_id)]
                        second_highest_buyer_bid = sorted(other_buyer_bids)[-2] if len(other_buyer_bids) > 1 else other_buyer_bids[0]
                        node.bid = (min_seller_price + second_highest_buyer_bid) / 2

            current_bids[node_id] = node.bid

        # Debugging output
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Checking for convergence...")

        # Check for convergence using dictionaries
        if has_converged(previous_bids, current_bids):
            converged = True
        
        previous_bids = current_bids.copy()
    
    return iteration  # Return the number of iterations to convergence


# Function to visualize market participation dynamics
def plot_market_dynamics(G_seller_buyer, iterations_data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    seller_ids = [node for node in G_seller_buyer.nodes if "Seller" in node]
    buyer_ids = [node for node in G_seller_buyer.nodes if "Buyer" in node]

    seller_id_to_num = {seller_id: idx for idx, seller_id in enumerate(seller_ids)}
    buyer_id_to_num = {buyer_id: idx for idx, buyer_id in enumerate(buyer_ids)}

    colors = plt.cm.tab20(np.linspace(0, 1, len(seller_ids)))

    for t, bids in enumerate(iterations_data):
        for buyer_id, sellers in bids.items():
            for seller_id in sellers:
                ax.scatter(t, buyer_id_to_num[buyer_id], seller_id_to_num[seller_id], 
                           color=colors[seller_id_to_num[seller_id]], s=50)

    ax.set_xlabel('Time Step (t)')
    ax.set_ylabel('Buyers')
    ax.set_zlabel('Sellers')

    ax.set_yticks(range(len(buyer_ids)))
    ax.set_yticklabels(buyer_ids)

    ax.set_zticks(range(len(seller_ids)))
    ax.set_zticklabels(seller_ids)

    plt.title("3D Visualization of Market Participation Dynamics")
    plt.show()

# Example usage
# iterations_data = [
#     {'Buyer_0': ['Seller_0', 'Seller_1'], 'Buyer_1': ['Seller_2']},
#     {'Buyer_0': ['Seller_0'], 'Buyer_2': ['Seller_1', 'Seller_2']},
#     {'Buyer_1': ['Seller_0'], 'Buyer_2': ['Seller_1']},
#     # Add more iterations data...
# ]

# plot_market_dynamics(G_seller_buyer, iterations_data)

# Function to run the simulation and compare convergence times
def main_compare_convergence():
    config = {
        "network_type": "random",
        "num_buyers": 20,
        "num_sellers": 5,
        "iterations": 20,
        "gamma": 1.0  # Elasticity parameter for buyer valuation
    }

    seller_config = {
        "seller_price_high": 100,
        "seller_price_low": 50,
        "seller_quantity_high": 20,
        "seller_quantity_low": 10
    }
    buyer_config = {
        "buyer_price_high": 50,
        "buyer_price_low": 20,
        "buyer_quantity_high": 20,
        "buyer_quantity_low": 10
    }

    sellers = [Node(f"Seller_{i}", price=np.random.uniform(seller_config["seller_price_low"], seller_config["seller_price_high"]),
                    quantity=-np.random.uniform(seller_config["seller_quantity_low"], seller_config["seller_quantity_high"])) for i in range(config["num_sellers"])]

    buyers = [Node(f"Buyer_{i}", price=np.random.uniform(buyer_config["buyer_price_low"], buyer_config["buyer_price_high"]),
                   quantity=np.random.uniform(buyer_config["buyer_quantity_low"], buyer_config["buyer_quantity_high"]), gamma=config["gamma"]) for i in range(config["num_buyers"])]

    G_seller_buyer = create_random_market_network(buyers, sellers)
    G_buyer_buyer = create_buyer_buyer_network(buyers, sellers, G_seller_buyer)
    G_seller_seller = nx.Graph()  # This can be used if you want to implement seller-seller connections

    # Run with opt-out
    iterations_with_opt_out = update_bids_with_utility_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config, use_opt_out=True)
    
    # Reset network and run without opt-out
    G_seller_buyer = create_random_market_network(buyers, sellers)
    G_buyer_buyer = create_buyer_buyer_network(buyers, sellers, G_seller_buyer)
    iterations_without_opt_out = update_bids_with_utility_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config, use_opt_out=False)
    
    # Compare results
    improvement_factor = iterations_without_opt_out / iterations_with_opt_out
    print(f"Convergence with opt-out: {iterations_with_opt_out} iterations")
    print(f"Convergence without opt-out: {iterations_without_opt_out} iterations")
    print(f"Improvement factor due to opt-out: {improvement_factor}")

if __name__ == "__main__":
    main_compare_convergence()