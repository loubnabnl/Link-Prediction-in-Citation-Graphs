import gzip
import pickle
import numpy as np
import scipy
import networkx as nx
from tqdm import tqdm
from random import randint
from sklearn.metrics import f1_score, log_loss

def load_features(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def get_training_graph(graph, edges_to_remove):
    res_graph = graph.copy()
    for edge in edges_to_remove:
        res_graph.remove_edge(edge[0], edge[1])
    return res_graph

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
    """Load adjacency matrix generated for the author graph"""

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

def sample_negative_links(G, neg_ratio):
    """create neg_ratio * G.number_of_edges negative samples (i.e pair of nodes without anedge)"""

    nodes = list(G.nodes())
    pos_edges = list(G.edges())
    n = G.number_of_nodes()
    m = G.number_of_edges()

    non_edges = []
    nb_neg_samples = neg_ratio * m

    for i in tqdm(range(int(nb_neg_samples))):
        n1 = nodes[randint(0, n-1)]
        n2 = nodes[randint(0, n-1)]
    
        while G.has_edge(n1, n2) or n1==n2 :
            n1 = nodes[randint(0, n-1)]
            n2 = nodes[randint(0, n-1)]
        non_edges.append((n1, n2))

    np.random.shuffle(pos_edges)
    np.random.shuffle(non_edges)

    # validation set samples
    number_pos_dev = int(0.1 * len(pos_edges))
    number_neg_dev = int(0.1 * len(non_edges))

    pos_samples_dev = pos_edges[:number_pos_dev]
    neg_samples_dev = non_edges[:number_neg_dev]
    pos_samples_train = list(set(pos_edges) - set(pos_samples_dev))
    neg_samples_train = list(set(non_edges) - set(neg_samples_dev))

    return pos_samples_train, neg_samples_train, pos_samples_dev, neg_samples_dev

def extract_features(graph, authors, n2v, t2v,a2v, samples, gd, path_adj , abstract_embedding='scibert'):
    """Build feature matrix from the graph as a concatenation of
    node embeddings, abstract embeddings and authors embeddings with other contextual and graph-based features

    Arguments:
        graph: the graph of the citation network
        authors:dictionnary of the lists of authors of papers
        n2v: node embeddings
        t2v: abstract embeddings
        a2v: author embeddings for the papers
        samples: node pairs of the data
        gd: graph dictionary with clustering coeff, and eigenvector centrality of graph
        path_adj: path to the adjacency matrix of the author graph we generated
        abstract_embedding: either 'scibert' or 'word2vec'
    Returns:
        numpy array storing features associated to node pairs in samples"""
        
    features = list()
    A, aut_to_index = load_adjacency_author(authors, path_adj = path_adj)

    for edge in tqdm(samples):

        ## Graph features
        sum_dg = graph.degree(edge[0]) + graph.degree(edge[1])
        diff_dg = abs(graph.degree(edge[0]) - graph.degree(edge[1]))
        AAI = list(nx.adamic_adar_index(graph, [(edge[0], edge[1])]))[0][2]
        JC = list(nx.jaccard_coefficient(graph, [(edge[0], edge[1])]))[0][2]
        PA = list(nx.preferential_attachment(graph, [(edge[0], edge[1])]))[0][2]
        CN = len(list(nx.common_neighbors(graph, u=edge[0], v=edge[1])))

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

        total_features = list(features_final) + [JC, AAI, PA, CN, cluster_coeff, eigenvector, cosine_node, cosine_abstract, cosine_author, 
                                                 sum_dg, diff_dg, common_authors, colab, colab_mean]

        features.append(total_features)

    return np.stack(features)