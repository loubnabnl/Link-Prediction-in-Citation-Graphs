import os
import gzip
import pickle
from tqdm import tqdm
import networkx as nx
from nodevectors import Node2Vec
import argparse

def create_node_embeddings(args):
    G = nx.read_edgelist(args.path_graph, delimiter=',', create_using=nx.Graph(), nodetype=int)
    nodes = list(G.nodes())
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    print("Training Node2vec...")
    model = Node2Vec(n_components=args.n_components, 
                walklen=40,
                epochs=30,
                threads=os.cpu_count())  

    model.fit(G)

    node_embeddings = dict()
    for node in tqdm(nodes):
        node_embeddings[node] = model.predict(node)

    print("----- Saving the Embeddings -----")
    file = gzip.GzipFile(args.path_save, 'wb')
    file.write(pickle.dumps(node_embeddings))
    file.close()

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pg", "--path_graph", type=str, default="../data/edgelist.txt", 
        help="Path to the graph edges text file")

    parser.add_argument("-nc", "--n_components", type=int, default=20,
                        help="Size of the embedding")
    parser.add_argument("-ps", "--path_save", type=str, default="../embeddings/node_embeddings1.emb",
                        help="Path to save the node embeddings file")
    

    create_node_embeddings(parser.parse_args())