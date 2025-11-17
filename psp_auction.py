import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np

# Mean-reverting process class for supply-to-demand ratio
class MeanRevertingProcess:
    def __init__(self, mean, speed, volatility):
        self.mean = mean
        self.speed = speed, 
        self.volatility = volatility
        self.value = mean
    
    def step(self):
        """Simulates a single time step in the mean-reverting process."""
        noise = np.random.normal(0, 1)
        self.value += self.speed * (self.mean - self.value) + self.volatility * noise
        return self.value

# Base Node class
class Node:
    def __init__(self, id, bid):
        self.id = id
        self.bid = bid
    
    def update_bid(self, neighbors_bids):
        """Base class doesn't implement this; to be implemented in subclasses."""
        pass

# Buyer class subclassing Node with flash memory
class Buyer(Node):
    def __init__(self, id, bid, sensitivity=0.5):
        super().__init__(id, bid)
        self.sensitivity = sensitivity  # How much the buyer is influenced by neighbors' bids
        self.last_bid_adjustment = 0  # Flash memory to remember last adjustment
    
    def update_bid(self, neighbors_bids):
        """Buyer updates bid based on neighbors' average bid and flash memory."""
        avg_bid = np.mean(neighbors_bids)
        # Calculate new bid with a component for flash memory
        new_bid = (1 - self.sensitivity) * self.bid + self.sensitivity * avg_bid + 0.1 * self.last_bid_adjustment
        
        # Update the flash memory with the change in bid
        self.last_bid_adjustment = new_bid - self.bid
        # Apply the new bid
        self.bid = new_bid

# Seller class subclassing Node
class Seller(Node):
    def __init__(self, id, ask_price, sensitivity=0.5):
        super().__init__(id, ask_price)
        self.sensitivity = sensitivity  # how much the seller is influenced by neighbors' bids
    
    def update_bid(self, neighbors_bids):
        """Seller updates ask price based on neighbors' average bid."""
        avg_bid = np.mean(neighbors_bids)
        self.bid = (1 - self.sensitivity) * self.bid + self.sensitivity * avg_bid

# Auction class with network-constrained matching
class ProgressiveSecondPriceAuctionWithConnectivity:
    def __init__(self, buyers, sellers, network, mean_reverting_process):
        self.buyers = buyers
        self.sellers = sellers
        self.network = network
        self.mean_reverting_process = mean_reverting_process
    
    def get_neighbors_bids(self, node):
        """Gets the bids from neighboring nodes."""
        neighbors = list(self.network.neighbors(node.id))
        neighbors_bids = [self.network.nodes[n]['node'].bid for n in neighbors]
        return neighbors_bids

    def is_connected(self, buyer_id, seller_id):
        """Check if a buyer and a seller are connected in the network."""
        return self.network.has_edge(buyer_id, seller_id)

    def run_auction(self):
        """Simulates one step of the auction with network-constrained matching."""
        # Update supply-demand ratio
        supply_demand_ratio = self.mean_reverting_process.step()

        # Update bids of all buyers and sellers based on neighbors
        for node in self.network.nodes():
            node_instance = self.network.nodes[node]['node']
            neighbors_bids = self.get_neighbors_bids(node_instance)
            if neighbors_bids:  # Update bid only if there are neighbors
                node_instance.update_bid(neighbors_bids)

        # Sort buyers and sellers by their bids
        buyers_bids = sorted(self.buyers, key=lambda x: x.bid, reverse=True)
        sellers_bids = sorted(self.sellers, key=lambda x: x.bid)

        matched_trades = []

        # Attempt to match buyers and sellers, considering only connected pairs
        for buyer in buyers_bids:
            for seller in sellers_bids:
                if self.is_connected(buyer.id, seller.id) and buyer.bid > seller.bid:
                    # Match if they are connected and the buyer's bid exceeds the seller's ask
                    matched_price = (buyer.bid + seller.bid) / 2  # Second-price auction logic
                    matched_trades.append((buyer.bid, seller.bid, matched_price))
                    sellers_bids.remove(seller)  # Remove seller from the pool once matched
                    break  # Move on to the next buyer once a match is found

        return matched_trades

# Create a sparser network
def create_sparse_buyer_seller_network(num_buyers, num_sellers, k=2):
    G = nx.Graph()

    # Add buyers and sellers as separate nodes
    buyers = [Buyer(id=i, bid=random.uniform(50, 100)) for i in range(num_buyers)]
    sellers = [Seller(id=i + num_buyers, ask_price=random.uniform(30, 80)) for i in range(num_sellers)]

    # Add edges only between buyers and sellers
    buyer_seller_connections = {}
    for buyer in buyers:
        G.add_node(buyer.id, node=buyer)
    for seller in sellers:
        G.add_node(seller.id, node=seller)

    # Create fewer connections between buyers and sellers (sparser network)
    for seller in sellers:
        # Connect each seller to fewer buyers (e.g., k=2 buyers per seller)
        connected_buyers = random.sample(buyers, k=min(k, len(buyers)))
        buyer_seller_connections[seller.id] = connected_buyers
        for buyer in connected_buyers:
            G.add_edge(seller.id, buyer.id)

    # Add connections between buyers who share the same seller
    for seller, buyer_list in buyer_seller_connections.items():
        for i, buyer_1 in enumerate(buyer_list):
            for j in range(i+1, len(buyer_list)):
                buyer_2 = buyer_list[j]
                if not G.has_edge(buyer_1.id, buyer_2.id):  # Avoid duplicate connections
                    G.add_edge(buyer_1.id, buyer_2.id)

    return G, buyers, sellers

# Simulation parameters
num_buyers = 10
num_sellers = 8
mean_reverting_process = MeanRevertingProcess(mean=1.0, speed=0.1, volatility=0.05)

# Create the sparser network
network_sparse, buyers_sparse, sellers_sparse = create_sparse_buyer_seller_network(num_buyers, num_sellers)

# Use the updated auction with network-constrained matching
auction_with_connectivity_sparse = ProgressiveSecondPriceAuctionWithConnectivity(buyers_sparse, sellers_sparse, network_sparse, mean_reverting_process)

# Track the bid adjustments for each buyer
price_adjustments_with_connectivity = {buyer.id: [] for buyer in buyers_sparse}

# Run the auction multiple times and track bid adjustments
for step in range(5):
    auction_with_connectivity_sparse.run_auction()
    for buyer in buyers_sparse:
        price_adjustments_with_connectivity[buyer.id].append(buyer.last_bid_adjustment)

# Plot both the sparse network and the price adjustments on the same figure with subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# First subplot: Sparse Buyer-Seller Network
axs[0].set_title("Sparse Buyer-Seller Network")
pos = nx.spring_layout(network_sparse)  # Generate positions for the nodes
nx.draw(network_sparse, pos=pos, ax=axs[0], node_color=['green' if node < num_buyers else 'red' for node in network_sparse.nodes], with_labels=True, node_size=500)
nx.draw_networkx_edges(network_sparse, pos, edgelist=[(u, v) for u, v in network_sparse.edges() if u >= num_buyers or v >= num_buyers], ax=axs[0], width=2, style='solid')
nx.draw_networkx_edges(network_sparse, pos, edgelist=[(u, v) for u, v in network_sparse.edges() if u < num_buyers and v < num_buyers], ax=axs[0], width=2, style='dotted')

# Second subplot: Price Adjustments Over Auction Steps
axs[1].set_title("Price Adjustments Over Auction Steps (Sparse Network)")
for buyer_id, adjustments in price_adjustments_with_connectivity.items():
    axs[1].plot(adjustments, label=f"Buyer {buyer_id}")
axs[1].set_xlabel("Auction Step")
axs[1].set_ylabel("Bid Adjustment")
axs[1].legend(title="Buyers")
axs[1].grid(True)

plt.tight_layout()
plt.show()