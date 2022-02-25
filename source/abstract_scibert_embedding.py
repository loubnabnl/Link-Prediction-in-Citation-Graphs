import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
import gzip
import argparse

def create_scibert_embeddings(args):
  """create embeddings for paper abstracts using pretrained SciBERT"""

  G = nx.read_edgelist(args.path_graph, delimiter=',', create_using=nx.Graph(), nodetype=int)
  nodes = list(G.nodes())
  n = G.number_of_nodes()
  m = G.number_of_edges()
  print('Number of nodes:', n)
  print('Number of edges:', m)

  # Read the abstract of each paper
  abstracts = dict()
  with open(args.abstracts, 'r',  encoding="utf8") as f:
      for line in f:
          node, abstract = line.split('|--|')
          abstracts[int(node)] = abstract


  #load model
  tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
  model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

  device = 'cuda' if torch.cuda.is_available() else 'cpu'  

  model.to(device)
  model.eval()

  text2vec = dict()

  print("Generating abstract embeddings...")
  for i in tqdm(range(len(nodes))):
    node_id = nodes[i]
    abstract = abstracts[node_id]

    tokens = tokenizer.encode(abstract, return_tensors='pt').to(device)

    token_embeddings = model(tokens)[0].detach().cpu().numpy()

    token_embeddings = token_embeddings.squeeze(0)
    
    text2vec[node_id] = token_embeddings
  print("----- Saving the Embeddings -----")
  file = gzip.GzipFile(args.path_save, 'wb')
  file.write(pickle.dumps(text2vec))
  file.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-pg", "--path_graph", type=str, default="data/edgelist.txt", 
        help="Path to the graph edges text file")
    parser.add_argument("-pa", "--abstracts", type=str, default="data/abstracts.txt", 
        help="Path to the abstracts text file")
    parser.add_argument("-ps", "--path_save", type=str, default="embeddings/abstracts_scibert_embeddings.emb",
                        help="Path to save the abstract embeddings file")

    create_scibert_embeddings(parser.parse_args())