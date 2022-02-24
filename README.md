# ALTEGRAD Link prediction challenge: 

Link prediction is the problem of predicting the existence of a link between two entities in a network. In this project, we worked on link prediction between scientific papers, the goal is to retrieve missing links in a citation network using text and graph-based features. This work was part of a challenge in the MVA ALTEGRAD course.

## Setup
``` bash
# Clone this repository
git clone https://github.com/loubnabnl/ALTEGRAD-challenge-Link-prediction.git
cd ALTEGRAD-challenge-Link-prediction/
# Install packages
pip install -r requirements.txt
```

## Data and embeddings
You can download the data (data.zip) [here](https://drive.google.com/drive/folders/1rQZR1iinXkCAvJwl1ETvxJUxGS2vEs4p?usp=sharing). It contains information about the edges of the network, as well as the abstracts and authors of the papers. You can also download the embeddings (embeddings.zip) we generated [here](https://drive.google.com/drive/folders/1rQZR1iinXkCAvJwl1ETvxJUxGS2vEs4p?usp=sharing). They must be respectively placed in the folders `data/` and `embeddings/`. Or you can just execute `./download.sh`.

## Execution
The folder `source/` contains scripts for generating node embeddings using Node2vec, abstract embeddings using two methods: SciBERT and Word2vec with a post-processing option to reduce the embedding size of SciBERT model. It also contains a script to generate author embeddings using NetMF on the nodes of an author graph we generate. The folder `models/`contains two classification models: MLP and XGBoost.

To execute the code you can run 
```
python main.py 
```

## To do

- [ ] Put data in drive, put embeddings too and change the naming
- [ ] Change paths in all files
- [ ] Make sure evrything runs correctly
- [ ] Add classification models to a folder  `model/`?
- [ ] Modify the notebook (to use only code in the repo)

Additional:
- [ ] Change code for extract_features and the way we read the data
- [ ] Write download.sh
- [ ] Put hyperparameters in config file