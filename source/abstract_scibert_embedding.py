import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
import gzip

EMBEDDING_FILENAME = './abstract_embeddings.emb'

# Create a graph
G = nx.read_edgelist('edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()
print('Number of nodes:', n)
print('Number of edges:', m)

# Read the abstract of each paper
abstracts = dict()
with open('abstracts.txt', 'r',  encoding="utf8") as f:
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

for i in tqdm(range(len(nodes))):
  node_id = nodes[i]
  abstract = abstracts[node_id]

  tokens = tokenizer.encode(abstract, return_tensors='pt').to(device)

  token_embeddings = model(tokens)[0].detach().cpu().numpy()

  token_embeddings = token_embeddings.squeeze(0)
  
  text2vec[node_id] = token_embeddings

file = gzip.GzipFile(EMBEDDING_FILENAME, 'wb')
file.write(pickle.dumps(text2vec))
file.close()
