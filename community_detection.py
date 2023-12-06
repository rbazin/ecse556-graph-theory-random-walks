from networkx.algorithms.community import greedy_modularity_communities, louvain_communities, girvan_newman
import networkx as nx
import joblib
from joblib import Parallel, delayed
import os
import random
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--algo", type=str, default=None, help="Community detection algorithm to use")
parser.add_argument("--output", type=str, default=None, help="Output file path")
parser.add_argument("--input", type=str, default=None, help="Input file path")

args = parser.parse_args()

intput_path = args.input
assert os.path.exists(intput_path), "Input file does not exist"


def calculate_betweenness(G, sampled_edges):
    subgraph = G.edge_subgraph(sampled_edges)
    centrality = nx.edge_betweenness_centrality(subgraph, normalized=True, weight='weight')
    return centrality

def parallel_betweenness_centrality(G, n_jobs=-1):
    all_edges = list(G.edges())
    random.shuffle(all_edges)
    sample_size = int(len(all_edges) / 10)
    sampled_edges = all_edges[:sample_size]

    centrality_values = Parallel(n_jobs=n_jobs)(delayed(calculate_betweenness)(G, sampled_edges) for _ in range(10))

    aggregate_centrality = {}
    for centrality in centrality_values:
        for edge, value in centrality.items():
            aggregate_centrality[edge] = aggregate_centrality.get(edge, 0) + value

    return max(aggregate_centrality, key=aggregate_centrality.get)

os.makedirs(os.path.dirname(args.output), exist_ok=True)

G = nx.read_weighted_edgelist(args.input)

algo = args.algo
assert algo in ["louvain", "girvan_newman", "greedy_modularity_communities"], "Invalid community detection algorithm"

if algo == "louvain":
    communities = list(louvain_communities(G))
elif algo == "girvan_newman":
    communities = list(girvan_newman(G, most_valuable_edge=parallel_betweenness_centrality))
elif algo == "greedy_modularity_communities":
    communities = list(greedy_modularity_communities(G))

joblib.dump(communities, os.path.join(args.output, f"./communities_{algo}.pkl"))