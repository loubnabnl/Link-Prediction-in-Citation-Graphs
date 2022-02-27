import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import pickle
import gzip
import argparse

from sklearn.decomposition import PCA

def dim_reduction(text2vec, first_size = 250, final_size=64):
  """method to reduce dimension of abstract embeddings in text2vec to a size=final_size
  source code: https://github.com/vyraun/Half-Size/blob/master/algo.py"""
  pca_embeddings = {}

  # PCA to get Top Components
  # first_size must be smaller than the mebdding size
  pca =  PCA(n_components = first_size)
  X = text2vec - np.mean(text2vec)
  X_fit = pca.fit_transform(X)
  U1 = pca.components_
  z = []

  # Removing Projections on Top Components
  for i, x in enumerate(X):
    for u in U1[0:7]:        
            x = x - np.dot(u.transpose(),x) * u 
    z.append(x)
  z = np.asarray(z)

  # PCA Dim Reduction
  pca =  PCA(n_components = final_size)
  X = z - np.mean(z)
  X = pca.fit_transform(X)

  # PCA to do Post-Processing Again
  pca =  PCA(n_components = final_size)
  X = X - np.mean(X)
  X = pca.fit_transform(X)
  Ufit = pca.components_

  X = X - np.mean(X)

  # Place embeddings in dictionnary
  text2vec_small = dict()
  keys = list(text2vec.keys())
  for i in range(len(keys)):
    text2vec_small[keys[i]] = X[i]
  return text2vec_small

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
  if args.dim_reduction:
        text2vec = dim_reduction(text2vec, final_size = args.reduction_size)
  print("----- Saving the Embeddings -----")
  file = gzip.GzipFile(args.path_save, 'wb')
  file.write(pickle.dumps(text2vec))
  file.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_graph", type=str, default="../data/edgelist.txt", 
        help="Path to the graph edges text file")
    parser.add_argument("--abstracts", type=str, default="../data/abstracts.txt", 
        help="Path to the abstracts text file")
    parser.add_argument("--dim_reduction", type=bool, default=False, 
        help="Argument scpecifying whether to keep the embedding size 768 of scibert or reduce it")
    parser.add_argument("--reduced_size", type=int, default=64, 
        help="embedding size after dimension reduction if applied")
    parser.add_argument("--path_save", type=str, default="../embeddings/abstract_embeddings.emb",
                        help="Path to save the abstract embeddings file")

    create_scibert_embeddings(parser.parse_args())