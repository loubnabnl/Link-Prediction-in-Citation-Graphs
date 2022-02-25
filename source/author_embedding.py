import os
import gzip
import pickle

import networkx as nx
import argparse
from time import time
import os
import karateclub
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix




def main(args):
    authors = dict()
    with open(args.authors, 'r',  encoding="utf8") as f:
        for line in f:
            node, author = line.split('|--|')
            authors[int(node)] = author
    G = nx.read_edgelist(args.path_graph, delimiter=',', create_using=nx.Graph(), nodetype=int)
    nodes = list(G.nodes())
    print("Starting creation of the embeddings")
    list_all_authors = []
    for x in list(authors.values()):
        x_list = x.split(",")
        x_list[-1] = x_list[-1][:-1]
        for auth in x_list:
            list_all_authors.append(auth)

    unique_authors = np.unique(list_all_authors)

    aut_to_index = {}
    index_to_aut = {}
    i=0
    for auth in unique_authors:
        aut_to_index[auth] = i
        index_to_aut[i] = auth
        i+=1
    print('Starting the creation of the adjacency matrix')
    n = len(unique_authors)
    A = csr_matrix((n, n), 
                          dtype = np.float32)
    for x in tqdm(list(authors.values())):
        x_list = x.split(",")
        x_list[-1] = x_list[-1][:-1]
        for i in range(len(x_list)):
            for j in range(i+1, len(x_list)):
                A[aut_to_index[x_list[i]], aut_to_index[x_list[j]]] +=1.0
                A[aut_to_index[x_list[j]], aut_to_index[x_list[i]]] +=1.0
    
    G_authors = nx.from_scipy_sparse_matrix(A)
    model = karateclub.NetMF(args.n_components)
    model.fit(G_authors)
    node_embedding = model.get_embedding()

    node_embeddings = dict()
    for node in tqdm(nodes):
        node_embeddings[node] = node_embedding[node]

    print("----- Saving the Embeddings -----")
    file = gzip.GzipFile(args.path_save, 'wb')
    file.write(pickle.dumps(node_embeddings))
    file.close()






if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-pg", "--path_graph", type=str, default="data/edgelist.txt", 
        help="Path to the graph edges text file")

    parser.add_argument("-nc", "--n_components", type=int, default=20,
                        help="Size of the embedding")
    parser.add_argument("-pa", "--authors", type=str, default="data/authors.txt", 
        help="Path to the author  text file")
    parser.add_argument("-ps", "--path_save", type=str, default="embeddings/authors_embeddings.emb",
                        help="Path to save the abstract embeddings file")
    

    main(parser.parse_args())
    
    
    
    
    