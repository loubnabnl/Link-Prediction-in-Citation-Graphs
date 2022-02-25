import os
import gzip
import pickle
from tqdm import tqdm
import networkx as nx
from nodevectors import Node2Vec


def create_node_embeddings(G, nodes, embedding_path, emb_size=128):
    model = Node2Vec(n_components=emb_size, 
                walklen=40,
                epochs=30,
                threads=os.cpu_count())  

    model.fit(G)

    node_embeddings = dict()
    for node in tqdm(nodes):
        node_embeddings[node] = model.predict(node)
        
    file = gzip.GzipFile(EMBEDDING_FILENAME, 'wb')
    file.write(pickle.dumps(node_embeddings))
    file.close()

if __name__ =="__main__":
    EMBEDDING_FILENAME = 'embeddings/node_embeddings.emb'

    G = nx.read_edgelist('data/edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
    nodes = list(G.nodes())
    n = G.number_of_nodes()
    m = G.number_of_edges()

    create_node_embeddings(G, nodes, EMBEDDING_FILENAME)