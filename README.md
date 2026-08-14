# MSSF
## Requirements
This project requires Python 3.9.19. Please create a virtual environment with the specified Python version. The environment and library requirements for running the project can be found in `requirements.txt`.



## Files
### data
This folder contains all input files needed by our model.

drug_mol.pkl: The word embedding matrix of drugs. We use Mol2vec model to learn the word embedding of drugs. Mol2vec can learn vector representations of molecular substructures pointing to similar directions of chemically related substructures. Each row of the matrix represents the word vector encoding of a drug.

glove_wordEmbedding.pkl: The word embedding matrix of side effects. We use the 300-dimensional Global Vectors (GloVe) trained on the Wikipedia dataset to represent the information of side effects. Each row of the matrix represents the word vector encoding of a side effect.

side_effect_semantic.pkl: The semantic similarity matrix of side effects. We download side effect descriptors from Adverse Drug Reaction Classification System and construct a novel model to calculate the semantic similarity of side effects. Each row of the matrix represents the similarity value between a side effect and all side effects in the benchmark dataset. The range of values is from 0 to 1.

Text_similarity_one.pkl, Text_similarity_two.pkl, Text_similarity_three.pkl, Text_similarity_four.pkl, Text_similarity_five.pkl: Five similarity matrices of drugs. These matrices are collected from the file "Chemical_chemical.links.detailed.v5.0.tsv.gz" in STITCH database. Each row of the matrices represents the similarity value between a drug and all drugs in the datasets respectively. The range of values is from 0 to 1.

fingerprint_similarity.pkl: The structure similarity matrix of the drugs. We calculate the structure similarities between drugs according to the Jaccard scores.

drug_target.pkl: The target protein information of the drugs is obtained from DrugBank database.

drug_pathway_similarity.pkl: The drug-pathway similarity matrix constructed from the benchmark dataset collected from DrugBank.

drug_pathway_enzyme_similarity.pkl: The drug–pathway-enzyme similarity matrix constructed from the benchmark dataset collected from DrugBank.

drug_side.pkl: The matrix has 757 rows and 994 columns to store the known drug-side effect frequency pairs. The element at the corresponding position of the matrix is set to the frequency value, otherwise 0.

### code
1.model.py: contains the network framework of our entire model.

2.mssf.py: test the predictive performance of our model under ten-fold cross-validation.

## Clone
The most recent code and data can be clone directly from Github with:
```bash
$ git clone https://github.com/dingxlcse/MSSF.git
```

## Running
After cloning the entire project from this repository, you can set up the environment based on the dependencies listed in requirements.txt. Then, run the following command to obtain the 10-fold cross-validation results of MSSF on the benchmark dataset:
```bash
$ python mssf.py
```

## Contact 
If you have any questions or suggestions with the code, please let us know. Contact Dingxi  Li at dingxlcse@csu.edu.cn
