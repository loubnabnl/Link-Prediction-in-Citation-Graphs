from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from utils import *
import numpy as np
import community.community_louvain as com
import pickle 
import csv
from xgboost import XGBClassifier
import argparse




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
    text2vec = load_features(args.abstract_path)
    aut2vec = load_features(args.authors_path)
    nodes2vec = load_features(args.nodes_path)
    
    
    pos_samples_train, neg_samples_train, pos_samples_dev, neg_samples_dev = sample_negative_links(G, neg_ratio=args.neg_ratio)
    
    graph = get_training_graph(G, pos_samples_dev)
    pr = nx.pagerank(graph,alpha=0.85, max_iter=200)
    graph_dicts = dict()
    graph_dicts["clustering_coeff"] = nx.algorithms.cluster.clustering(graph)
    graph_dicts["eigenvector"] = nx.algorithms.centrality.eigenvector_centrality(graph)
    partition = com.best_partition(graph)
    
        #split data
    train_samples = pos_samples_train + neg_samples_train
    train_labels = [1 for x in pos_samples_train] + [0 for x in neg_samples_train]
    dev_samples = pos_samples_dev + neg_samples_dev
    dev_labels = [1 for x in pos_samples_dev] + [0 for x in neg_samples_dev]
    
    

    adjacency_author = np.load('./adjacencyfinal.npz')

    filename = 'aut_to_indexfinal.pkl'
    with open(filename, 'rb') as f:
        aut_to_index  = pickle.load(f)
        
        
    X_train = extract_features(graph, authors, nodes2vec, text2vec,aut2vec, train_samples, pr, graph_dicts, partition)
    X_dev = extract_features(graph, authors, nodes2vec, text2vec,aut2vec, dev_samples, pr, graph_dicts, partition)
    if args.model == "xgboost":
        clf = XGBClassifier(max_depth=4, scale_pos_weight=3, learning_rate=0.1, n_estimators=2000, n_jobs=4, tree_method='gpu_hist', predictor="gpu_predictor", random_state=42, seed=42)
        clf.fit(X_train, list(train_labels), eval_metric="logloss", early_stopping_rounds=300, eval_set=[(X_dev, list(dev_labels))], verbose=1)
    elif args.model == "MLP":
        
        clf_mlp = MLPClassifier(hidden_layer_sizes=(150,100,60,30,2), verbose=1, early_stopping=True, n_iter_no_change=3, max_iter = 10)
        clf_mlp.fit(X_train, train_labels)
        y_pred = clf.predict_proba(X_dev[:, -25:])[:,1]
        score = return_metrics(dev_labels, y_pred)
        print("Validation LogLoss: ", score[2])
        
    elif args.model == "LogisticRegression":
        clf = LogisticRegression()
        clf.fit(X_train, train_labels)
        y_pred = clf.predict_proba(X_dev[:, -25:])[:,1]
        score = return_metrics(dev_labels, y_pred)
        print("Validation LogLoss: ", score[2])
        
    elif args.model == "kerasMLP":
        clf = neural_net(X_train, train_labels, X_dev, dev_labels)
        X_test = extract_features(graph, authors, nodes2vec, text2vec,aut2vec, node_pairs, pr, graph_dicts, partition)
        y_pred = clf.predict(X_test)

    node_pairs = list()
    with open('test.txt', 'r') as f:
        for line in f:
            t = line.split(',')
            node_pairs.append((int(t[0]), int(t[1])))
    if not(X_test):
        X_test = extract_features(graph, authors, nodes2vec, text2vec,aut2vec, node_pairs, pr, graph_dicts, partition)
        
        y_pred = clf.predict_proba(X_test)[:,1]
        
    predictions = zip(range(len(y_pred)), y_pred)
    with open(f"/content/submission.csv","w") as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(['id','predicted'])
        for row in predictions:
            csv_out.writerow(row)
    
    
if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-pg", "--path_graph", type=str, default="edgelist.txt", 
        help="Path to the graph edges text file")
    parser.add_argument("-ab", "--abstract_path", type=str, default="../abstract_embeddings.emb",
                        help="Path to the abstract text file")

    parser.add_argument("-pn", "--nodes_path", type=str, default="../NETMF42_emb.emb", 
        help="Path to the node embeddings file")
    parser.add_argument("-pa", "--authors_path", type=str, default="../authors_emb.emb", 
        help="Path to the author embeddings file")
    parser.add_argument("-a", "--authors", type=str, default="../authors.txt", 
        help="Path to the author  text file")
    parser.add_argument("-nr", "--neg_ratio", type=int, default=1, 
        help="Path to the author  text file")
    parser.add_argument("-m", "--model", type=str, default="xgboost", 
        help="Model to compute the predictions")
    
    
    
    

    main(parser.parse_args())    