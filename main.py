import heapq
from collections import defaultdict
import random
import time
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class TransportationOptimizer:
    """
    Transportation optimizer using bidirectional Dijkstra algorithm 
    with contraction hierarchies for efficient routing similar to Google Maps.
    """
    
    def __init__(self):
        # Graph representation: node -> [(neighbor, weight, edge_id)]
        self.graph = defaultdict(list)
        self.reversed_graph = defaultdict(list)
        self.nodes = set()
        self.edge_count = 0
        self.node_levels = {}  # Stores importance level for each node
        self.shortcuts = {}  # (u, v) -> [(via_node, original_edge1, original_edge2), ...]
        
    def add_edge(self, u, v, weight, bidirectional=True):
        """Add an edge to the graph with optional bidirectionality."""
        self.nodes.add(u)
        self.nodes.add(v)
        
        edge_id = self.edge_count
        self.edge_count += 1
        
        self.graph[u].append((v, weight, edge_id))
        self.reversed_graph[v].append((u, weight, edge_id))
        
        if bidirectional:
            edge_id = self.edge_count
            self.edge_count += 1
            self.graph[v].append((u, weight, edge_id))
            self.reversed_graph[u].append((v, weight, edge_id))
            
    def add_road_network(self, edges):
        """
        Add a list of edges representing a road network.
        Each edge is (node1, node2, weight, bidirectional).
        """
        for u, v, weight, bidirectional in edges:
            self.add_edge(u, v, weight, bidirectional)
    
    def _calculate_node_importance(self, node):
        """
        Calculate importance of a node based on:
        1. Edge difference (number of shortcuts needed - number of edges removed)
        2. Number of contracted neighbors
        3. Number of edges
        """
        # Count incoming and outgoing edges
        in_edges = len(self.reversed_graph[node])
        out_edges = len(self.graph[node])
        
        # Count contracted neighbors
        contracted_neighbors = sum(1 for n, _, _ in self.graph[node] if n in self.node_levels)
        contracted_neighbors += sum(1 for n, _, _ in self.reversed_graph[node] if n in self.node_levels)
        
        # Calculate shortcut cover
        shortcut_count = 0
        edge_count = in_edges + out_edges
        
        # For each pair of incoming and outgoing edges
        for in_node, in_weight, _ in self.reversed_graph[node]:
            for out_node, out_weight, _ in self.graph[node]:
                if in_node != out_node:  # Avoid loops
                    # This would create a shortcut
                    shortcut_count += 1
        
        # Edge difference: shortcut_count - edge_count
        # Lower is better for contraction
        edge_diff = shortcut_count - edge_count
        
        # Combine factors (weights can be adjusted)
        importance = edge_diff * 5 + contracted_neighbors * 2 + edge_count
        
        return importance
    
    def preprocess_contraction_hierarchies(self, status_func=None):
        """
        Preprocess the graph to build contraction hierarchies.
        This is a core optimization technique used in modern routing systems.
        """
        if status_func:
            status_func("Preprocessing contraction hierarchies...")
        
        start_time = time.time()
        
        # Initialize node importances
        node_importances = {}
        for node in self.nodes:
            node_importances[node] = self._calculate_node_importance(node)
        
        # Priority queue for nodes to contract
        pq = [(node_importances[node], node) for node in self.nodes]
        heapq.heapify(pq)
        
        level = 0
        processed_count = 0
        total_nodes = len(self.nodes)
        
        while pq:
            _, node = heapq.heappop(pq)
            
            # Skip if already processed
            if node in self.node_levels:
                continue
            
            # Contract node
            self._contract_node(node, level)
            level += 1
            
            processed_count += 1
            if processed_count % max(1, total_nodes // 10) == 0:
                percentage = (processed_count / total_nodes) * 100
                if status_func:
                    status_func(f"Processed {processed_count}/{total_nodes} nodes ({percentage:.1f}%)...")
            
            # Update importance of affected nodes
            for neighbor, _, _ in list(self.graph[node]) + list(self.reversed_graph[node]):
                if neighbor not in self.node_levels:  # Skip contracted nodes
                    old_importance = node_importances.get(neighbor, float('inf'))
                    new_importance = self._calculate_node_importance(neighbor)
                    
                    if new_importance != old_importance:
                        node_importances[neighbor] = new_importance
                        heapq.heappush(pq, (new_importance, neighbor))
        
        if status_func:
            status_func(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
            status_func(f"Created {len(self.shortcuts)} shortcuts")
    
    def _contract_node(self, node, level):
        """Contract a node by adding shortcuts between its neighbors."""
        self.node_levels[node] = level
        
        # Find all pairs of incoming and outgoing edges
        for in_node, in_weight, in_edge in self.reversed_graph[node]:
            if in_node in self.node_levels:  # Skip already contracted nodes
                continue
                
            for out_node, out_weight, out_edge in self.graph[node]:
                if out_node in self.node_levels or in_node == out_node:  # Skip contracted nodes and self-loops
                    continue
                
                # Calculate shortcut weight
                shortcut_weight = in_weight + out_weight
                
                # Check if shortcut is necessary (witness search)
                if self._is_shortcut_necessary(in_node, out_node, shortcut_weight, node):
                    # Add shortcut to the graph
                    self.add_edge(in_node, out_node, shortcut_weight)
                    
                    # Store original edges that this shortcut represents
                    shortcut_key = (in_node, out_node)
                    if shortcut_key not in self.shortcuts:
                        self.shortcuts[shortcut_key] = []
                    self.shortcuts[shortcut_key].append((node, in_edge, out_edge))
    
    def _is_shortcut_necessary(self, source, target, shortcut_weight, excluded_node, max_hops=5):
        """
        Determine if a shortcut is necessary by running a limited local search.
        If there's no alternative path that's shorter or equal, the shortcut is necessary.
        """
        # Run a limited Dijkstra search to see if there's a better path
        dist = {source: 0}
        visited = set()
        pq = [(0, source, 0)]  # (distance, node, hop count)
        
        while pq:
            distance, node, hops = heapq.heappop(pq)
            
            if node == target:
                return distance > shortcut_weight  # Shortcut is necessary if no better path found
            
            if node in visited or hops >= max_hops:
                continue
                
            visited.add(node)
            
            for neighbor, weight, _ in self.graph[node]:
                if neighbor != excluded_node and neighbor not in visited:
                    new_dist = distance + weight
                    if new_dist < dist.get(neighbor, float('inf')):
                        dist[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor, hops + 1))
        
        # If target not reached, shortcut is necessary
        return True
    
    def bidirectional_dijkstra(self, source, target):
        """
        Bidirectional Dijkstra's algorithm using contraction hierarchies.
        This is similar to the core routing algorithm used by Google Maps.
        """
        if source == target:
            return [source], 0
        
        # Forward search (from source)
        forward_dist = {source: 0}
        forward_prev = {}
        forward_pq = [(0, source)]
        forward_settled = set()
        
        # Backward search (from target)
        backward_dist = {target: 0}
        backward_prev = {}
        backward_pq = [(0, target)]
        backward_settled = set()
        
        # Best meeting point and distance
        best_meeting_node = None
        best_path_length = float('inf')
        
        # Search until both priority queues are empty
        while forward_pq and backward_pq:
            # Exit if the smallest distance in either direction exceeds the best path
            if forward_pq[0][0] + backward_pq[0][0] >= best_path_length:
                break
            
            # Forward search step
            current_dist, current_node = heapq.heappop(forward_pq)
            
            if current_node in forward_settled:
                continue
                
            forward_settled.add(current_node)
            
            # Check if current node is in backward search space
            if current_node in backward_dist:
                path_length = current_dist + backward_dist[current_node]
                if path_length < best_path_length:
                    best_path_length = path_length
                    best_meeting_node = current_node
            
            # Only expand to higher level nodes (upward search in CH)
            for neighbor, weight, _ in self.graph[current_node]:
                # In CH, only consider edges to higher-level nodes
                if neighbor not in self.node_levels or (
                    current_node in self.node_levels and 
                    self.node_levels[neighbor] > self.node_levels[current_node]
                ):
                    new_dist = current_dist + weight
                    if new_dist < forward_dist.get(neighbor, float('inf')):
                        forward_dist[neighbor] = new_dist
                        forward_prev[neighbor] = current_node
                        heapq.heappush(forward_pq, (new_dist, neighbor))
            
            # Backward search step
            current_dist, current_node = heapq.heappop(backward_pq)
            
            if current_node in backward_settled:
                continue
                
            backward_settled.add(current_node)
            
            # Check if current node is in forward search space
            if current_node in forward_dist:
                path_length = current_dist + forward_dist[current_node]
                if path_length < best_path_length:
                    best_path_length = path_length
                    best_meeting_node = current_node
            
            # Only expand to higher level nodes (upward search in CH)
            for neighbor, weight, _ in self.reversed_graph[current_node]:
                # In CH, only consider edges to higher-level nodes
                if neighbor not in self.node_levels or (
                    current_node in self.node_levels and 
                    self.node_levels[neighbor] > self.node_levels[current_node]
                ):
                    new_dist = current_dist + weight
                    if new_dist < backward_dist.get(neighbor, float('inf')):
                        backward_dist[neighbor] = new_dist
                        backward_prev[neighbor] = current_node
                        heapq.heappush(backward_pq, (new_dist, neighbor))
        
        # Track nodes explored for visualization
        explored_nodes = forward_settled.union(backward_settled)
        
        # No path found
        if best_meeting_node is None:
            return None, float('inf'), explored_nodes
        
        # Reconstruct the path
        path = self._reconstruct_hierarchical_path(source, target, best_meeting_node, forward_prev, backward_prev)
        
        return path, best_path_length, explored_nodes
    
    def _reconstruct_hierarchical_path(self, source, target, meeting_node, forward_prev, backward_prev):
        """Reconstruct the complete path from the bidirectional search."""
        # First, build the up-path from source to meeting node
        up_path = []
        current = meeting_node
        while current != source:
            up_path.append(current)
            current = forward_prev[current]
        up_path.append(source)
        up_path.reverse()
        
        # Then, build the down-path from meeting node to target
        down_path = []
        current = backward_prev[meeting_node]
        while current is not None:
            down_path.append(current)
            if current == target:
                break
            current = backward_prev.get(current)
        
        # Combine up-path and down-path
        full_path = up_path + down_path
        
        # Unpack shortcuts to get the actual path
        return self._unpack_shortcuts(full_path)
    
    def _unpack_shortcuts(self, packed_path):
        """Unpack shortcuts to get the actual path with all intermediate nodes."""
        if len(packed_path) <= 1:
            return packed_path
            
        result = [packed_path[0]]
        
        for i in range(len(packed_path) - 1):
            u, v = packed_path[i], packed_path[i+1]
            
            # Check if (u, v) is a shortcut
            if (u, v) in self.shortcuts:
                # Get one of the possible shortcuts
                shortcut = self.shortcuts[(u, v)][0]
                middle_node = shortcut[0]
                
                # Recursively unpack
                expanded = self._unpack_shortcuts([u, middle_node, v])
                result.extend(expanded[1:])
            else:
                # Regular edge
                result.append(v)
        
        return result
    
    def find_shortest_path(self, source, target, status_func=None):
        """Find the shortest path using bidirectional Dijkstra with contraction hierarchies."""
        start_time = time.time()
        path, distance, explored_nodes = self.bidirectional_dijkstra(source, target)
        end_time = time.time()
        
        if path is None:
            if status_func:
                status_func(f"No path found from {source} to {target}")
            return None, float('inf'), 0, explored_nodes
            
        if status_func:
            status_func(f"Path found in {(end_time - start_time) * 1000:.2f}ms")
        return path, distance, end_time - start_time, explored_nodes

    def standard_dijkstra(self, source, target):
        """Standard Dijkstra algorithm for comparison."""
        start_time = time.time()
        
        dist = {source: 0}
        prev = {}
        visited = set()
        pq = [(0, source)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current == target:
                break
                
            if current in visited:
                continue
                
            visited.add(current)
            
            for neighbor, weight, _ in self.graph[current]:
                new_dist = current_dist + weight
                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    prev[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        # Reconstruct path
        if target not in prev and target != source:
            return None, float('inf'), time.time() - start_time, visited
            
        path = []
        current = target
        while current != source:
            path.append(current)
            current = prev[current]
        path.append(source)
        path.reverse()
        
        end_time = time.time()
        return path, dist.get(target, float('inf')), end_time - start_time, visited


def create_grid_network(width, height, weight_range=(1, 10), one_way_prob=0.1, diagonal_prob=0.3):
    """Create a grid network of nodes with random weights."""
    edges = []
    for i in range(width):
        for j in range(height):
            node = (i, j)
            
            # Connect to right neighbor
            if i < width - 1:
                weight = random.randint(*weight_range)
                bidirectional = random.random() > one_way_prob
                edges.append((node, (i+1, j), weight, bidirectional))
                
            # Connect to bottom neighbor
            if j < height - 1:
                weight = random.randint(*weight_range)
                bidirectional = random.random() > one_way_prob
                edges.append((node, (i, j+1), weight, bidirectional))
                
            # Add some diagonals for more realistic road networks
            if i < width - 1 and j < height - 1 and random.random() < diagonal_prob:
                weight = random.randint(*weight_range)
                weight = int(weight * 1.4)  # Diagonals are longer
                bidirectional = random.random() > one_way_prob
                edges.append((node, (i+1, j+1), weight, bidirectional))
    
    return edges


def create_real_world_network(num_nodes=100, avg_connections=4, area_size=10, weight_factor=10):
    """Create a more realistic road network with random node positions."""
    edges = []
    
    # Generate random node positions
    node_positions = {}
    for i in range(num_nodes):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        node_positions[i] = (x, y)
    
    # Create a distance matrix
    distances = {}
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            x1, y1 = node_positions[i]
            x2, y2 = node_positions[j]
            dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
            distances[(i, j)] = dist
    
    # Sort distances and create edges for closest pairs
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    
    # Create enough edges to have an average of avg_connections per node
    target_edges = num_nodes * avg_connections // 2
    
    for (i, j), dist in sorted_distances[:target_edges]:
        # Convert distance to weight (travel time)
        weight = int(dist * weight_factor)
        bidirectional = random.random() > 0.1  # 10% chance of one-way streets
        edges.append((i, j, weight, bidirectional))
    
    return edges, node_positions


def visualize_network(G, pos, path=None, explored=None, source=None, target=None, title="Road Network"):
    """Visualize the road network and optionally a path."""
    plt.figure(figsize=(10, 10))
    
    # Draw the basic network
    edge_color = 'lightgray'
    node_color = 'lightblue'
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color=edge_color)
    
    # Draw explored nodes if provided
    if explored:
        nx.draw_networkx_nodes(G, pos, nodelist=list(explored), 
                               node_color='lightyellow', alpha=0.7, node_size=80)
    
    # Draw path if provided
    if path:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                               edge_color='red', width=2)
        # Highlight path nodes
        nx.draw_networkx_nodes(G, pos, nodelist=path, 
                               node_color='orange', node_size=80)
    
    # Highlight source and target
    if source:
        nx.draw_networkx_nodes(G, pos, nodelist=[source], 
                               node_color='green', node_size=150)
    if target:
        nx.draw_networkx_nodes(G, pos, nodelist=[target], 
                               node_color='red', node_size=150)
    
    # Draw all nodes (on top)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_color, alpha=0.6)
    
    # Add labels for source and target
    if source and target:
        labels = {source: "Start", target: "End"}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight='bold')
    
    plt.title(title)
    plt.axis('off')
    return plt


def run_streamlit_app():
    st.set_page_config(page_title="Transportation Route Optimizer", layout="wide")
    
    st.title("Transportation Route Optimizer")
    st.markdown("""
    This app demonstrates a transportation optimization algorithm similar to what powers Google Maps.
    It uses **Bidirectional Dijkstra** with **Contraction Hierarchies** for efficient routing.
    """)
    
    # Initialize session state variables
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
        st.session_state.network_type = None
        st.session_state.network_params = None
        st.session_state.network_generated = False
        st.session_state.G = None
        st.session_state.pos = None
        st.session_state.status_messages = []
    
    # Sidebar for configuration
    st.sidebar.header("Network Configuration")
    
    network_type = st.sidebar.radio(
        "Network Type",
        ["Grid Network", "Real-World Network"]
    )
    
    if network_type == "Grid Network":
        col1, col2 = st.sidebar.columns(2)
        width = col1.slider("Grid Width", 5, 100, 15)
        height = col2.slider("Grid Height", 5, 100, 15)
        min_weight = col1.slider("Min Travel Time", 1, 20, 5)
        max_weight = col2.slider("Max Travel Time", min_weight, 50, 20)
        diagonal_prob = st.sidebar.slider("Diagonal Road Probability", 0.0, 1.0, 0.3)
        one_way_prob = st.sidebar.slider("One-Way Road Probability", 0.0, 0.5, 0.1)
    else:  # Real-World Network
        num_nodes = st.sidebar.slider("Number of Nodes", 20, 200, 80)
        avg_connections = st.sidebar.slider("Avg Connections per Node", 2, 8, 4)
        area_size = st.sidebar.slider("Area Size", 5, 20, 10)
        weight_factor = st.sidebar.slider("Travel Time Factor", 1, 20, 10)
    
    # Button to generate network
    if st.sidebar.button("Generate New Network"):
        st.session_state.network_generated = False
    
    # Algorithm selection
    algorithm = st.sidebar.radio(
        "Routing Algorithm",
        ["Bidirectional Dijkstra with Contraction Hierarchies", "Standard Dijkstra"]
    )
    
    # Check if network parameters changed
    current_params = {
        'network_type': network_type,
        'algorithm': algorithm
    }
    
    if network_type == "Grid Network":
        current_params.update({
            'width': width,
            'height': height,
            'min_weight': min_weight,
            'max_weight': max_weight,
            'diagonal_prob': diagonal_prob,
            'one_way_prob': one_way_prob
        })
    else:
        current_params.update({
            'num_nodes': num_nodes,
            'avg_connections': avg_connections,
            'area_size': area_size,
            'weight_factor': weight_factor
        })
    
    # Check if we need to regenerate the network
    regenerate = False
    if not st.session_state.network_generated:
        regenerate = True
    elif st.session_state.network_type != network_type:
        regenerate = True
    elif st.session_state.network_params != current_params:
        regenerate = True
    
    # Generate the network if needed
    if regenerate:
        with st.spinner("Generating network..."):
            st.session_state.optimizer = TransportationOptimizer()
            st.session_state.status_messages = []
            
            # Status update function
            def update_status(message):
                st.session_state.status_messages.append(message)
            
            if network_type == "Grid Network":
                # Create grid network
                road_network = create_grid_network(
                    width, height, 
                    weight_range=(min_weight, max_weight),
                    diagonal_prob=diagonal_prob,
                    one_way_prob=one_way_prob
                )
                st.session_state.optimizer.add_road_network(road_network)
                
                # Create positions for visualization
                pos = {(i, j): (i, j) for i in range(width) for j in range(height)}
                
                # Create NetworkX graph for visualization
                G = nx.DiGraph()
                for u, v, weight, bidirectional in road_network:
                    G.add_edge(u, v, weight=weight)
                    if bidirectional:
                        G.add_edge(v, u, weight=weight)
                
            else:  # Real-World Network
                # Create realistic network
                road_network, pos = create_real_world_network(
                    num_nodes=num_nodes, 
                    avg_connections=avg_connections,
                    area_size=area_size,
                    weight_factor=weight_factor
                )
                st.session_state.optimizer.add_road_network(road_network)
                
                # Create NetworkX graph for visualization
                G = nx.DiGraph()
                for u, v, weight, bidirectional in road_network:
                    G.add_edge(u, v, weight=weight)
                    if bidirectional:
                        G.add_edge(v, u, weight=weight)
            
            # Store in session state
            st.session_state.G = G
            st.session_state.pos = pos
            st.session_state.network_type = network_type
            st.session_state.network_params = current_params
            
            # Preprocess if using contraction hierarchies
            if algorithm == "Bidirectional Dijkstra with Contraction Hierarchies":
                st.session_state.optimizer.preprocess_contraction_hierarchies(update_status)
            
            st.session_state.network_generated = True
    
    # Display status messages
    with st.expander("Preprocessing Log", expanded=False):
        for msg in st.session_state.status_messages:
            st.text(msg)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display the network visualization
        if st.session_state.G and st.session_state.pos:
            fig = visualize_network(st.session_state.G, st.session_state.pos)
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.subheader("Route Planning")
        
        # Source and destination selection
        if network_type == "Grid Network":
            # For grid, use coordinates
            st.markdown("#### Select Start Point")
            start_col1, start_col2 = st.columns(2)
            start_x = start_col1.number_input("Start X", 0, width-1, 0, key="start_x")
            start_y = start_col2.number_input("Start Y", 0, height-1, 0, key="start_y")
            source = (int(start_x), int(start_y))
            
            st.markdown("#### Select End Point")
            end_col1, end_col2 = st.columns(2)
            end_x = end_col1.number_input("End X", 0, width-1, width-1, key="end_x")
            end_y = end_col2.number_input("End Y", 0, height-1, height-1, key="end_y")
            target = (int(end_x), int(end_y))
        else:
            # For real-world, use node IDs
            source = st.number_input("Start Node ID", 0, num_nodes-1, 0, key="start_node")
            target = st.number_input("End Node ID", 0, num_nodes-1, num_nodes-1, key="end_node")
        
        # Find route button
        if st.button("Find Route"):
            if source == target:
                st.warning("Start and end points are the same!")
            else:
                with st.spinner("Finding route..."):
                    # Status update function
                    route_status = st.empty()
                    
                    def update_route_status(message):
                        route_status.text(message)
                    
                    # Choose algorithm
                    if algorithm == "Bidirectional Dijkstra with Contraction Hierarchies":
                        path, distance, time_taken, explored = st.session_state.optimizer.find_shortest_path(
                            source, target, update_route_status
                        )
                    else:
                        path, distance, time_taken, explored = st.session_state.optimizer.standard_dijkstra(
                            source, target
                        )
                    
                    if path:
                        st.success(f"Route found in {time_taken*1000:.2f}ms")
                        st.info(f"Travel time: {distance} minutes")
                        st.info(f"Explored {len(explored)} nodes ({len(explored)/len(st.session_state.optimizer.nodes)*100:.1f}% of network)")
                        
                        # Display path visualization
                        fig = visualize_network(
                            st.session_state.G, 
                            st.session_state.pos, 
                            path=path, 
                            explored=explored,
                            source=source, 
                            target=target, 
                            title=f"Route: {distance} units, {len(path)} nodes"
                        )
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.error("No route found between these points!")
    
    # Explanation section
    with st.expander("How It Works", expanded=False):
        st.markdown("""
        ### How This Route Optimizer Works
        
        This application demonstrates advanced routing algorithms similar to those used by Google Maps:
        
        #### Bidirectional Dijkstra with Contraction Hierarchies
        
        1. **Preprocessing (Contraction Hierarchies)**:
           - Assigns importance levels to each node in the network
           - Contracts nodes in order of importance, adding shortcuts between neighbors
           - Creates a hierarchical structure that allows extremely fast queries
        
        2. **Bidirectional Search**:
           - Searches simultaneously from both start and end points
           - Uses the hierarchy to only consider relevant edges
           - Stops when the search frontiers meet
        
        3. **Path Reconstruction**:
           - Combines the forward and backward paths
           - Expands any shortcuts to show the complete route
           - Calculates total travel time
        
        #### Performance Benefits
        
        - **Contraction Hierarchies** can make queries 100-1000x faster than standard Dijkstra
        - **Bidirectional search** further reduces the search space
        - The preprocessing step is done once, enabling fast repeated queries
        
        #### Real-World Applications
        
        This same technology powers:
        - Google Maps routing
        - Uber/Lyft driver dispatching
        - Logistics and delivery route optimization
        - Emergency response routing
        """)

    # Add performance comparison section
    with st.expander("Performance Comparison", expanded=False):
        st.markdown("""
        ### Algorithm Performance Characteristics
        
        | Algorithm | Preprocessing Time | Query Time | Memory Usage | Best For |
        |-----------|--------------------|------------|--------------|----------|
        | Standard Dijkstra | None | O(E + V log V) | O(V) | Small networks |
        | Bidirectional Dijkstra | None | O(E + V log V) | O(V) | Medium networks |
        | Contraction Hierarchies | O(V log V) | O(log V) | O(E) | Large networks, many queries |
        
        *V = number of nodes, E = number of edges*
        """)

        if st.button("Run Performance Test"):
            with st.spinner("Running performance tests..."):
                # Select test nodes
                nodes = list(st.session_state.optimizer.nodes)
                test_pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(5)]
                
                # Test standard Dijkstra
                std_times = []
                for s, t in test_pairs:
                    _, _, time_taken, _ = st.session_state.optimizer.standard_dijkstra(s, t)
                    std_times.append(time_taken)
                
                # Test CH Dijkstra
                ch_times = []
                for s, t in test_pairs:
                    _, _, time_taken, _ = st.session_state.optimizer.find_shortest_path(s, t)
                    ch_times.append(time_taken)
                
                # Display results
                col1, col2 = st.columns(2)
                col1.metric("Average Standard Dijkstra Time", f"{np.mean(std_times)*1000:.2f} ms")
                col2.metric("Average CH Dijkstra Time", f"{np.mean(ch_times)*1000:.2f} ms")
                
                st.write("Speedup Factor:", f"{np.mean(std_times)/np.mean(ch_times):.1f}x")

# Run the app
if __name__ == "__main__":
    run_streamlit_app()