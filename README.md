# Dynamic GNNs on Heterogeneous Citation Networks

### Overview
This project implements and evaluates several popular Graph Neural Networks (GNNs) — **Graph Convolutional Networks (GCN)**, **Graph Attention Networks (GAT)**, and **GraphSAGE** on four benchmark citation graph datasets: Cora, PubMed, Citeseer, and DBLP. A core feature of this implementation is the use of dynamic, adaptive graph partitioning. This technique breaks down large graphs into smaller subgraphs, enabling efficient training on resource-constrained hardware while mitigating the challenges posed by graph size and imbalance. The partitioning strategy is fully dynamic, periodically re-partitioning the graph during training to ensure balanced subgraph sizes and optimize model performance.

---

### Key Features 
- **Multiple GNN Implementations**: The repository includes modular implementations of GCN, GAT, and GraphSAGE, allowing for easy experimentation and comparison.

- **Adaptive Graph Partitioning**: Uses METIS via the DGL library to dynamically partition large graphs. The partitioning scheme is adaptive, automatically re-partitioning the graph during training if significant load imbalance is detected across partitions.

- **Minibatch Training**: Utilizes NeighborLoader from PyTorch Geometric to perform efficient minibatch training on the dynamically partitioned graphs.

- **Comprehensive Dataset Support**: The code is designed to work seamlessly with the Cora, PubMed, Citeseer, and DBLP datasets, all handled through the Planetoid dataset in PyTorch Geometric.

- **Performance Evaluation**: Includes functions to evaluate the trained models on the test splits of each dataset, providing insights into accuracy and efficiency.

---

### GNN Architectures 
- **GCN (Graph Convolutional Network)**: A simple and effective model that propagates node features across the graph using localized spectral convolutions. The implementation includes batch normalization to stabilize training.

- **GAT (Graph Attention Network)**: This model extends GCN by incorporating a self-attention mechanism. It learns the importance of different neighbors for each node, allowing it to aggregate information more selectively.

- **GraphSAGE (SAmple and aggreGatE)**: A popular framework for inductive representation learning. Instead of using the full graph, it learns a function that aggregates features from a fixed number of sampled neighbors, making it highly scalable.

---

### Dataset Overview 
The project leverages four widely used academic datasets for node classification tasks.

- **Cora**: A dataset of research papers where nodes represent papers and edges represent citations. 

- **Citeseer**: Similar to Cora, this dataset consists of scientific publications.

- **PubMed**: A larger dataset of biomedical publications.

- **DBLP**: A co-authorship network where nodes are authors, and edges represent co-authorship.

---

### Dependencies 

- torch
- torch_geometric
- dgl (Deep Graph Library)
- pyg_lib
- tqdm 
