from joblib import Parallel, delayed, dump
from networkx.algorithms.community import girvan_newman
import networkx as nx

import argparse
import logging
import os
from time import time

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(level)s - %(message)s")


def compute_girvan_newman(graph_path):
    graph_name = os.path.basename(graph_path)
    size = int(graph_name.split("_")[2])

    logging.info(f"Starting job for graph of size: {size}")

    subgraph = nx.read_weighted_edgelist(graph_path)
    start = time()
    girvan_clusters = list(girvan_newman(subgraph))
    end = time()

    logging.info(f"Finished job for graph of size: {size}, time taken: {end - start}")

    return {
        "size": size,
        "time": end - start,
        "clusters": girvan_clusters,
        "graph_name": graph_name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_dir", type=str, help="Input graph dir path")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    assert os.path.exists(args.graphs_dir), "Input graph dir does not exist"

    graphs_paths = [
        os.path.join(args.graphs_dir, file_name)
        for file_name in os.listdir(args.graphs_dir)
        if file_name.startswith("cleaned_graph_") and file_name.endswith("_nodes.edge")
    ]

    logging.info(f"Computing Girvan-Newman for {len(graphs_paths)} graphs")

    res_dicts = Parallel(n_jobs=-1)(
        delayed(compute_girvan_newman)(graph_path) for graph_path in graphs_paths
    )

    dump(res_dicts, args.output)


if __name__ == "__main__":
    main()
