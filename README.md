# ALTEGRAD Link prediction challenge: 

Link prediction is the problem of predicting the existence of a link between two entities in a network. In this project, we worked on link prediction between scientific papers, the goal is to retrieve missing links in a citation network using text and graph-based features. This work was part of a challenge in the MVA ALTEGRAD course.

## Setup
``` bash
# Clone this repository
git clone https://github.com/loubnabnl/Link-Prediction-in-Citation-Graphs.git
cd Link-Prediction-in-Citation-Graphs/
# Install packages
pip install -r requirements.txt
```

## Data and embeddings
You can download the data [here](https://drive.google.com/drive/folders/1rQZR1iinXkCAvJwl1ETvxJUxGS2vEs4p?usp=sharing). It contains information about the edges of the network, as well as the abstracts and authors of the papers. You can also download the embeddings we generated [here](https://drive.google.com/drive/folders/1rQZR1iinXkCAvJwl1ETvxJUxGS2vEs4p?usp=sharing). They must be respectively placed in the folders `data/` and `embeddings/`. 

## Execution
The folder `source/` contains scripts for generating node embeddings using Node2vec, abstract embeddings using two methods: SciBERT and Word2vec with a post-processing option to reduce the embedding size of SciBERT model. It also contains a script to generate author embeddings using NetMF on the nodes of an author graph we generate. The folder `models/`contain an MLP classification model, the other model: XGBoost, is direclty defined in `main.py`. 

To execute the code, using our embeddings, you can run 
```
python main.py 
```
