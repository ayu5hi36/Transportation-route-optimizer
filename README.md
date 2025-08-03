# 🚗 Transportation Route Optimizer

An interactive route planning tool for optimizing paths in transportation networks, built using advanced graph algorithms like **Bidirectional Dijkstra** and **Contraction Hierarchies**. This project showcases how graph theory techniques can be used for fast and scalable routing in real-world applications.

---

## ✨ Features

- 🔄 **Bidirectional Dijkstra’s Algorithm** for faster shortest path queries
- 🏗️ **Contraction Hierarchies** for ultra-fast pathfinding in large graphs
- 📈 **Interactive Visualization** using NetworkX and Matplotlib
- 🆚 **Performance Comparison** with standard Dijkstra’s algorithm
- 🌐 **Streamlit-based Web App** for user-friendly experimentation

---

## 🧠 Core Algorithms

### 📍 Standard Dijkstra’s Algorithm
- Finds shortest path from a source to all nodes
- Time Complexity: `O(E + V log V)`
- Used as a performance baseline

### 📍 Bidirectional Dijkstra
- Runs simultaneous forward and backward searches
- Stops early when both searches meet
- Faster than standard Dijkstra

### 📍 Contraction Hierarchies (CH)
- Preprocesses the graph to assign node importance
- Contracts less important nodes and adds shortcuts
- Enables fast hierarchical search with unpacking
- **Query Time:** `O(log V)`

---

## 🗺️ Graph Model

- **Nodes** → Locations/Intersections
- **Edges** → Roads (weighted by travel time)
- Supports both **one-way** and **bidirectional** roads
- Graph stored using adjacency lists (forward + reverse)

---

## 📊 Performance

| Algorithm                    | Preprocessing Time | Query Time | Best For             |
|-----------------------------|--------------------|------------|----------------------|
| Standard Dijkstra           | None               | High       | Small networks       |
| Bidirectional Dijkstra      | None               | Medium     | Medium networks      |
| CH + Bidirectional Dijkstra | Moderate           | Very Low   | Large-scale routing  |

---

## 🖼️ UI Preview (Streamlit)

- Choose between **Grid** or **Real-World** road network generation
- Tune parameters: node count, weights, directionality, etc.
- Select start and end points for route planning
- Visualize shortest path and nodes explored
- Compare performance metrics of both algorithms

---


