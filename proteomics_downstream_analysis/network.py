import matplotlib.pyplot as plt
import networkx as nx
from community import community_louvain
from collections import defaultdict
import math
from sklearn.cluster import SpectralClustering


class NetworkAnalysis:

    def __init__(self):
        pass

    # SOME FUNCTIONS TO CREATE A NETWORK

    def create_protein_network(self, ntwrk_coef, ntwrk_pval, coef_threshold=0.5, pval_threshold=0.01):
        G = nx.Graph()
        for pair, coef in ntwrk_coef.items():
            pval = ntwrk_pval[pair]
            if abs(coef) >= coef_threshold and pval <= pval_threshold:
                prot_a, prot_b = pair.split('_')
                G.add_edge(prot_a, prot_b, weight=abs(coef))
        return G

    def cluster_network(self, G, method='louvain', seed=4, resolution=1):
        if method == 'louvain':
            partition = community_louvain.best_partition(G, random_state=seed, resolution=resolution)
        elif method == 'spectral':
            adj_matrix = nx.to_numpy_array(G)
            n_clusters = min(10, G.number_of_nodes())  # Adjust number of clusters based on network size
            sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=100, random_state=seed)
            labels = sc.fit_predict(adj_matrix)
            partition = {node: label for node, label in zip(G.nodes(), labels)}
        else:
            raise ValueError("Unsupported clustering method")
        return partition

    def analyze_clusters(self, G, partition, return_cluster=False):
        clusters = {}
        for node, cluster_id in partition.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node)
        
        print("Cluster Analysis:")
        for cluster_id, proteins in clusters.items():
            print(f"Cluster {cluster_id}:")
            print(f"  Number of proteins: {len(proteins)}")
            print(f"  Example proteins: {', '.join(proteins[:5])}")
            
            # Calculate and print average degree within cluster
            subgraph = G.subgraph(proteins)
            avg_degree = sum(dict(subgraph.degree()).values()) / len(subgraph)
            print(f"  Average degree within cluster: {avg_degree:.2f}")
            print()

        if return_cluster:
            return clusters

    def identify_hub_proteins(self, G, top_n=50, method='degree'):
        if method == 'degree':
            centrality = nx.degree_centrality(G)
        elif method == 'betweenness':
            centrality = nx.betweenness_centrality(G)
        elif method == 'eigenvector':
            centrality = nx.eigenvector_centrality(G)
        else:
            raise ValueError("Unsupported centrality method")
        
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_centrality[:top_n])

    def visualize_network_with_hubs(self, G, partition, hubs):
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
        
        # Highlight hub proteins
        nx.draw_networkx_nodes(G, pos, nodelist=hubs.keys(), node_color='yellow', node_size=100)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        
        # Add labels for hub proteins
        nx.draw_networkx_labels(G, pos, {node: node for node in hubs.keys()}, font_size=8)
        
        plt.title("Protein Interaction Network with Clusters and Hub Proteins")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def protein_network_analysis(self, ntwrk_coef, ntwrk_pval, coef_threshold=0.5, pval_threshold=0.05, clustering_method='louvain', hub_method='degree', top_n_hubs=10):
        # Create network
        G = self.create_protein_network(ntwrk_coef, ntwrk_pval, coef_threshold, pval_threshold)
        print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        # Cluster network
        partition = self.cluster_network(G, method=clustering_method)
        print(f"Network clustered using {clustering_method} method")

        # Identify hub proteins
        hubs = self.identify_hub_proteins(G, top_n=top_n_hubs, method=hub_method)
        print(f"Top {top_n_hubs} hub proteins identified using {hub_method} centrality:")
        for protein, score in hubs.items():
            print(f"{protein}: {score:.4f}")

        # Visualize network with hubs
        self.visualize_network_with_hubs(G, partition, hubs)

        # Analyze clusters
        self.analyze_clusters(G, partition)

        return G, partition, hubs


    def spread_layout(self, G, k=None, iterations=50):
        """
        Custom layout function to spread nodes.
        
        Args:
        G (networkx.Graph): The graph to layout
        k (float): Optimal distance between nodes
        iterations (int): Number of iterations to run the algorithm
        
        Returns:
        dict: A dictionary of node positions
        """
        if k is None:
            k = 1 / math.sqrt(len(G))
        
        pos = nx.spring_layout(G, k=k*2, iterations=iterations)
        return pos


    def spread_layoutv2(self, G, k=None, iterations=50):
        if k is None:
            k = 1 / math.sqrt(len(G))
        pos = nx.spring_layout(G, k=k*2, iterations=iterations)
        return pos

    def spread_hubs_layout(self, G, centrality_measure='betweenness', k=2, iterations=50, scale=1):
        """
        Custom layout function to spread out hub nodes.
        
        Args:
        G (networkx.Graph): The graph to layout
        centrality_measure (str): Centrality measure to use ('betweenness' or 'degree')
        k (float): Node repulsion strength
        iterations (int): Number of iterations for the spring layout
        scale (float): Scaling factor for node spreading
        
        Returns:
        dict: A dictionary of node positions
        """
        if centrality_measure == 'betweenness':
            centrality = nx.betweenness_centrality(G)
        elif centrality_measure == 'degree':
            centrality = nx.degree_centrality(G)
        else:
            raise ValueError("Invalid centrality measure. Choose 'betweenness' or 'degree'.")
        
        # Initial layout
        pos = nx.spring_layout(G, k=k, iterations=iterations, seed=42)
        
        # Spread out high centrality nodes
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        for i, (node, _) in enumerate(sorted_nodes[:10]):  # Adjust top 10 nodes
            angle = 2 * math.pi * i / 10
            r = scale * math.sqrt(centrality[node])
            pos[node] = (r * math.cos(angle), r * math.sin(angle))
        
        return pos

    def visualize_protein_clusters(self, G, partition, max_clusters=9, min_cluster_size=10, 
                                centrality_measure='betweenness', top_n_hubs=5,
                                custom_proteins=None, save_path=False):
        """
        Visualize separate clusters from a protein network with custom coloring and annotation
        for both hub proteins and custom proteins.
        
        Args:
        G (networkx.Graph): The full protein network
        partition (dict): A dictionary mapping nodes to their cluster ids
        max_clusters (int): Maximum number of clusters to visualize
        min_cluster_size (int): Minimum size of cluster to visualize
        centrality_measure (str): Centrality measure to use ('betweenness' or 'degree')
        top_n_hubs (int): Number of top hub proteins to annotate
        custom_proteins (list): List of specific proteins to color differently and annotate
        
        Returns:
        None (displays plots)
        """
        clusters = defaultdict(list)
        for node, cluster_id in partition.items():
            clusters[cluster_id].append(node)
        
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        filtered_clusters = [cluster for cluster in sorted_clusters if len(cluster[1]) >= min_cluster_size][:max_clusters]
        
        grid_size = math.ceil(math.sqrt(len(filtered_clusters)))
        fig = plt.figure(figsize=(5*grid_size, 5*grid_size))
        
        for i, (cluster_id, nodes) in enumerate(filtered_clusters):
            subgraph = G.subgraph(nodes)
            ax = fig.add_subplot(grid_size, grid_size, i+1)
            
            pos = self.spread_hubs_layout(subgraph, centrality_measure=centrality_measure, k=2, iterations=50, scale=0.3)
            
            if centrality_measure == 'betweenness':
                centrality = nx.betweenness_centrality(subgraph)
            else:
                centrality = nx.degree_centrality(subgraph)
            
            # Identify hub proteins
            hub_proteins = set(dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n_hubs]).keys())
            
            # Prepare node colors and sizes
            node_colors = []
            node_sizes = []
            for node in subgraph.nodes():
                if node in hub_proteins:
                    node_colors.append('red')
                    node_sizes.append(600)
                elif custom_proteins and node in custom_proteins:
                    node_colors.append('yellow')
                    node_sizes.append(450)
                else:
                    node_colors.append('lightblue')
                    node_sizes.append(300)
            
            nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(subgraph, pos, edge_color='lightgrey', alpha=0.5, ax=ax, width=0.5)
            
            # Annotate both hub proteins and custom proteins
            labels = {}
            for node in subgraph.nodes():
                # if node in hub_proteins or (custom_proteins and node in custom_proteins):
                    labels[node] = node
            
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)
            
            ax.set_title(f"Cluster {cluster_id}\n(Size: {len(nodes)})")
            ax.axis('off')
            
            print(f"Cluster {cluster_id}:")
            print(f"Number of proteins: {len(nodes)}")
            print(f"Hub proteins: {', '.join(hub_proteins)}")
            if custom_proteins:
                custom_in_cluster = set(custom_proteins) & set(nodes)
                print(f"Custom proteins in cluster: {', '.join(custom_in_cluster)}")
            print(f"Average {centrality_measure} centrality: {sum(centrality.values()) / len(subgraph):.4f}")
            print("\n")
        
        plt.tight_layout()

        if save_path:
            # Ensure the directory exists
            plt.savefig(save_path, transparent=True, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
