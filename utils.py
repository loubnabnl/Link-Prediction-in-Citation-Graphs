import gzip
import pickle
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, roc_auc_score

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
    preds = torch.cat(preds, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()
    preds_label = np.where(preds > thres, 1, 0)
    f1 = f1_score(true, preds_label)
    auc = roc_auc_score(true, preds)
    return f1, auc

def extract_features(graph, authors, n2v, t2v, samples, paper_sim, node_mapping, pagerank, katz):
    feature_func = lambda x, y: np.concatenate([x, y])

    features = list()

    for edge in tqdm(samples):
        ## Graph features
        
        node_left, node_right = edge[0], edge[1]
        # Retrieve features of node2vec embedding
        diff_n2v = feature_func(n2v[node_left], n2v[node_right])

        # Resource Allocation Index
        RAI = list(nx.resource_allocation_index(graph, [(node_left, node_right)]))[0][2]
        # Jaccard Coefficient
        JC = list(nx.jaccard_coefficient(graph, [(node_left, node_right)]))[0][2]
        # Adamic Adar Index
        AAI = list(nx.adamic_adar_index(graph, [(node_left, node_right)]))[0][2]
        # Preferential Attachment
        PA = list(nx.preferential_attachment(graph, [(node_left, node_right)]))[0][2]
        # Common Neighbors
        CN = len(list(nx.common_neighbors(graph, u=node_left, v=node_right)))
        # Page Rank
        PR = np.log(pagerank[node_left] * pagerank[node_right])
        # Katz
        KZ = np.log(katz[node_left] * katz[node_right])

        graph_features = list(diff_n2v) + [PR, KZ, RAI, JC, AAI, PA, CN]

        ## Text features

        # Retrieve cosine similarity between asbtract
        cos_sim = paper_sim[node_mapping[node_left], node_mapping[node_right]]

        # Common Authors
        authors_left = authors[node_left]
        authors_right = authors[node_right]

        if authors_left is None or authors_right is None:
            common_authors = float('nan')
        else:
            common_authors = len(list(set(authors_left).intersection(authors_right)))

        text_features = [cos_sim, common_authors]

        total_features = graph_features + text_features

        features.append(total_features)

    return features

def get_training_graph(graph, edges_to_remove):
    res_graph = graph.copy()
    for edge in edges_to_remove:
        res_graph.remove_edge(edge[0], edge[1])
    return res_graph
