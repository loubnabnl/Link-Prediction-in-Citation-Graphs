import numpy as np
import networkx as nx
import pickle 
import csv
from xgboost import XGBClassifier
import argparse
from utils import load_features, sample_negative_links, get_training_graph, extract_features, return_metrics
from models import train_MLP


def main(args):
    #load authors
    authors = dict()
    with open(args.authors, 'r',  encoding="utf8") as f:
        for line in f:
            node, author = line.split('|--|')
            authors[int(node)] = author

    # Create a graph
    G = nx.read_edgelist(args.path_graph, delimiter=',', create_using=nx.Graph(), nodetype=int)
    nodes = list(G.nodes())
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Features related to the texts
    print('Loading embeddings...')
    text2vec = load_features(args.abstract_path)
    aut2vec = load_features(args.authors_path)
    nodes2vec = load_features(args.nodes_path)
    
    pos_samples_train, neg_samples_train, pos_samples_dev, neg_samples_dev = sample_negative_links(G, 
                                                                    args.neg_ratio, test_node_pairs)
    
    graph = get_training_graph(G, pos_samples_dev)
    graph_dicts = dict()
    graph_dicts["clustering_coeff"] = nx.algorithms.cluster.clustering(graph)
    graph_dicts["eigenvector"] = nx.algorithms.centrality.eigenvector_centrality(graph)
    
    #training and validation data
    train_samples = pos_samples_train + neg_samples_train
    train_labels = [1 for x in pos_samples_train] + [0 for x in neg_samples_train]
    dev_samples = pos_samples_dev + neg_samples_dev
    dev_labels = [1 for x in pos_samples_dev] + [0 for x in neg_samples_dev]
    
    adjacency_author = np.load(args.authors_adj)
        
    print('Generating training and validation features...')  
    X_train = extract_features(graph, authors, nodes2vec, text2vec,aut2vec, train_samples, graph_dicts, path_adj=args.authors_adj)
    X_dev = extract_features(graph, authors, nodes2vec, text2vec,aut2vec, dev_samples, graph_dicts, path_adj=args.authors_adj)
    
    #classification
    print(f'Training Classifier {args.model} ...')
    if args.model == "xgboost":
        clf = XGBClassifier(max_depth=4, scale_pos_weight=3, learning_rate=0.1, n_estimators=2000, n_jobs=4, tree_method='gpu_hist', predictor="gpu_predictor", random_state=42, seed=42)
        clf.fit(X_train, list(train_labels), eval_metric="logloss", early_stopping_rounds=300, eval_set=[(X_dev, list(dev_labels))], verbose=1)
    
    elif args.model == "MLP":
        clf = train_MLP(X_train, train_labels, X_dev, dev_labels)
    
    else:
        raise ValueError('Invalid classifcation model name')
    
    print("Test phase...")
    test_node_pairs = list()
    with open(args.path_test, 'r') as f:
        for line in f:
            t = line.split(',')
            test_node_pairs.append((int(t[0]), int(t[1])))
    
    if not(X_test):
        X_test = extract_features(graph, authors, nodes2vec, text2vec, aut2vec, test_node_pairs, graph_dicts, path_adj=args.authors_adj)
        
        y_pred = clf.predict_proba(X_test)[:,1]
        
    predictions = zip(range(len(y_pred)), y_pred)
    with open(f"submission.csv","w") as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(['id','predicted'])
        for row in predictions:
            csv_out.writerow(row)
    
    
if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-pg", "--path_graph", type=str, default="data/edgelist.txt", 
        help="Path to the graph edges text file")
    parser.add_argument("-a", "--authors", type=str, default="data/authors.txt", 
        help="Path to the author  text file")
    parser.add_argument("-pg", "--path_test", type=str, default="data/test.txt", 
        help="Path to the test file")
    parser.add_argument("-ab", "--abstract_path", type=str, default="embeddings/abstract_embeddings.emb",
                        help="Path to the abstract text file")
    parser.add_argument("-pn", "--nodes_path", type=str, default="embeddings/node_embeddings.emb", 
        help="Path to the node embeddings file")
    parser.add_argument("-pa", "--authors_path", type=str, default="embeddings/authors_embeddings.emb", 
        help="Path to the author embeddings file")
    parser.add_argument("-adj", "--authors_adj", type=str, default="embeddings/adjacencyfinal.npz", 
        help="Path to the author graph adjacency matrix")
    parser.add_argument("-nr", "--neg_ratio", type=int, default=1, 
        help="ratio of negative samples")
    parser.add_argument("-m", "--model", type=str, default="xgboost", 
        help="Model to compute the predictions")

    main(parser.parse_args())    