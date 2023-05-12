# Predicting-Druggable-Proteins
Predicting Druggable Proteins from Primary Amino Acid Sequence
Druggable proteins are those that can interact with drug-like molecules and are used as targets for drugs that can potentially be effective in treating disease. Predicting the druggability of proteins is therefore a crucial step in the development of new drugs. Computational models that can predict the druggability of proteins using only their primary sequence is a time and cost-effective complement to the more precise yet more time consuming and laborious experimental methods that analyze the tertiary (3D) structure of proteins. In here, we will explore the use of machine learning methods for predicting the druggablity of proteins from their primary amino acid sequence.

First, we use following properties to extract features.

- RSpolar
- DPC
- RSsecond
- AAC
- RScharge

After feature extraction we develop machine leraning models to predict the drugrability of the sequence.

# How to Use

- First install all the requirements by running "python -m pip install -r requirements.txt"
- Put required 4 files in the directory
    - TR_POS
    - TR_NEG
    - TS_POS
    - TS_NEG
- Give executable permission by running "chmod +x run.sh"
- Finally run using "./run.sh positive_training_data negative_training_data positive_testing_data negative_testing_data"
    - Use FASA format
    - Default is "./run.sh TR_pos_SPIDER.fasta TR_neg_SPIDER.fasta TS_pos_SPIDER.fasta TS_neg_SPIDER.fasta"

# Group Details

This is a project done for CS4742 - Bioinformatics by

- By Group : Deoxyribos
- Members : GUNATHILAKA C.G. 180211X, WIMALARATNE G.D.S.U. 180718H, KARUNASENA S.T. 180313L, ASHKAR M.H.M. 180048D

#


