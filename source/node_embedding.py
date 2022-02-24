import os
import gzip
import pickle
from tqdm import tqdm

import networkx as nx
from nodevectors import Node2Vec


EMBEDDING_FILENAME = './node_embeddings.emb'

# Create a graph
G = nx.read_edgelist('edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()

model = Node2Vec(n_components=128, 
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