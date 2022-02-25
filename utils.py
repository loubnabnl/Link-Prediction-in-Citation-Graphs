import gzip
import pickle
import numpy as np
import scipy
import torch
import networkx as nx
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, log_loss

def load_features(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
def cosine_similarity(node_ids, text2vec):
    text_embedding = []
    for node in node_ids:
        text_embedding.append(text2vec[node])

    text_embedding = np.stack(text_embedding, axis=0)
    similarity = cosine_similarity(text_embedding, text_embedding)

    return similarity

def return_metrics(true, preds, thres=0.5):
    preds_label = np.where(preds > thres, 1, 0)
    f1 = f1_score(true, preds_label)
    logloss = log_loss(true, preds.astype(np.float64))
    return f1, logloss

def cosine_sim(arr1, arr2, eps=0.001):
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 < eps or norm2 < eps:
      return 0
    return np.dot(arr1, arr2)/(np.linalg.norm(arr1)*np.linalg.norm(arr2))

def load_adjacency_author(authors, path_adj = 'data/adjacencyfinal.npz'):
  A = scipy.sparse.load_npz(path_adj)
  list_all_authors = []
  aut = list(authors.values())
  for x in aut:
    x_list = x.split(",")
    x_list[-1] = x_list[-1][:-1]
    for auth in x_list:
      list_all_authors.append(auth)
  unique_authors = np.unique(list_all_authors)

  aut_to_index = {}
  i=0
  for auth in unique_authors:
    aut_to_index[auth] = i
    i+=1
  return A, aut_to_index

def extract_features(graph, authors, n2v, t2v,a2v, samples, gd, partition, abstract_embedding='scibert'):
    
    features = list()
    A, aut_to_index = load_adjacency_author(authors, path_adj = 'adjacencyfinal.npz')

    for edge in tqdm(samples):

        ## Graph features
        sum_dg = graph.degree(edge[0]) + graph.degree(edge[1])
        diff_dg = abs(graph.degree(edge[0]) - graph.degree(edge[1]))
        AAI = list(nx.adamic_adar_index(graph, [(edge[0], edge[1])]))[0][2]
        JC = list(nx.jaccard_coefficient(graph, [(edge[0], edge[1])]))[0][2]
        PA = list(nx.preferential_attachment(graph, [(edge[0], edge[1])]))[0][2]
        CN = len(list(nx.common_neighbors(graph, u=edge[0], v=edge[1])))
        if partition[edge[0]] == partition[edge[1]]:
            com_partition = 1
        else:
            com_partition = 0
        cluster_coeff = gd["clustering_coeff"][edge[0]] * gd["clustering_coeff"][edge[1]]
        eigenvector = gd["eigenvector"][edge[0]] * gd["eigenvector"][edge[1]]

        ## Embeddings
        cosine_node = cosine_sim(n2v[edge[0]], n2v[edge[1]])
        cosine_author = cosine_sim(a2v[edge[0]], a2v[edge[1]])
        if abstract_embedding=='scibert':
            cosine_abstract = cosine_sim(t2v[edge[0]], t2v[edge[1]])
        else:
            #compute L2 distance for distance word2vec sentence embeddings
            cosine_abstract = np.linalg.norm(t2v[edge[0]] - t2v[edge[1]])
        
        features_final = np.concatenate([n2v[edge[0]], n2v[edge[1]], t2v[edge[0]], t2v[edge[1]], 
                                        a2v[edge[0]], a2v[edge[1]]])
        
        ## More features
        # Common Authors
        authors_left = authors[edge[0]]
        authors_right = authors[edge[1]]

        # Collaboration measure
        L1 = list(set(authors_left.strip().split(',')))
        L2 = list(set(authors_right.strip().split(',')))
        colab = 0
        for author in L1:
          for author2 in L2:
            colab += A[aut_to_index[author], aut_to_index[author2]] 

        colab_mean = colab/(len(authors_left)*len(authors_right))

        if authors_left is None or authors_right is None:
            common_authors = float('nan')
        else:
            common_authors = len(list(set(authors_left.strip().split(',')).intersection(authors_right.strip().split(','))))

        total_features = list(features_final) + [JC, AAI, PA, CN, com_partition, cluster_coeff, eigenvector, cosine_node, dist_abstract, cosine_author, 
                                                 sum_dg, diff_dg, common_authors, colab, colab_mean]

        features.append(total_features)

    return np.stack(features)