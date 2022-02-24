import networkx as nx
from gensim.models import word2vec
import nltk
import pickle
import gzip
import argparse

def abstract_encoding(args):
    # Create a graph
    G = nx.read_edgelist(args.path_graph, delimiter=',', create_using=nx.Graph(), nodetype=int)
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print('Number of nodes:', n)
    print('Number of edges:', m)

    # Read the abstract of each paper
    abstracts = dict()
    with open(args.path_abstract, 'r',  encoding="utf8") as f:
        for line in f:
            node, abstract = line.split('|--|')
            abstracts[int(node)] = abstract
    return G, abstracts

def abstract_embedding(abstract, model):
    num_features = model.trainables.layer1_size
    result = np.zeros(num_features)
    words = abstract.split()
    oov = 0
    for word in words:
        if word in model.wv.vocab:
            result += model.wv[word]
        else:
            oov += 1
    if len(words) - oov != 0:
        result /= (len(words) - oov)
    else:
        result = 0
    return result

def main(args):
    G, abstracts = abstract_encoding(args)
    nodes = list(G.nodes())
    EMBEDDING_FILENAME = args.path_save
    
    # Tokenization
    nltk.download('punkt')
    stemmer = nltk.stem.PorterStemmer()
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words('english'))

    #Word training corpus
    print('Generating Vocabulary...')
    training_words = []
    for node in nodes:
      training_words += [word for word in abstracts[node].split() if word not in stpwds]
    training_words = list(set(training_words))


    model = word2vec.Word2Vec(training_words, workers=4, size=64, min_count=5, window=20)

    abstract_emb = dict()
    for node in tqdm(nodes):
      abstract_emb[node] = abstract_embedding(abstracts[node], model)


    #@title
    print('saving embeddings')
    file = gzip.GzipFile(args.path_save, 'wb')
    file.write(pickle.dumps(EMBEDDING_FILENAME))
    file.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-pg", "--path_graph", type=str, default="edgelist.txt", 
        help="Path to the graph edges text file")
    parser.add_argument("-pa", "--path_abstract", type=str, default="../abstracts.txt",
                        help="Path to the abstract text file")
    parser.add_argument("-ps", "--path_save", type=str, default="./abstract_wv_embeddings.emb",
                        help="Path to save the abstract embeddings file")

    

    main(parser.parse_args())