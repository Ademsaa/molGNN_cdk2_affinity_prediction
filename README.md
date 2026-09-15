# CDK2 Kinase Inhibitor Activity Prediction

This project builds a small computational drug discovery pipeline for predicting the potency (pIC50) of CDK2 kinase inhibitors, going from raw bioactivity data collection all the way to graph neural network models. It was built as a learning project while working through cheminformatics and molecular machine learning concepts, so the notebooks are written step by step rather than as a polished package.

## What this covers

The Cyclin-Dependent Kinase 2 (CDK2) is a well studied cancer drug target. Because a CDK2-only dataset from ChEMBL is fairly small (around 2000 compounds), the pipeline also pulls in related kinases (CDK1, CDK4, CDK6, CDK9) and treats them as auxiliary tasks, since many CDK inhibitors are tested against several family members at once and share structural patterns.

The end goal is to train models that take a molecule (as a SMILES string) and predict its pIC50 against CDK2, then compare a few different modeling approaches on the same held out test set.

## Pipeline

The notebooks are numbered in the order they should be run.

1. `1.compound_data_aquisation.ipynb`
   Connects to ChEMBL through `chembl_webresource_client`, pulls bioactivity records for CDK1/2/4/6/9 (IC50, binding assays only, human targets), cleans up units and duplicates, and computes pIC50 values. Ends with roughly 7000 compound-kinase activity records.

2. `2.unwanted_filtering.ipynb`
   Applies Lipinski's rule of five and removes PAINS and Brenk structural alerts. Includes a radar plot comparing the dataset's molecular weight, H-bond donors/acceptors and LogP against standard drug-like thresholds. Around 6500 of the original 7000 compounds pass the rule of five.


3.`3.gnn.ipynb` / `4.gnn.ipynb` (later versions of the same notebook)
   Builds PyTorch Geometric graph objects from the SMILES strings (atom and bond level features), splits the data using a scaffold split rather than a random split so the test set contains chemotypes not seen during training, then trains and compares two GNN architectures:
   - a dual encoder model that fuses a GAT branch and a GIN branch with the precomputed fingerprints (referred to as AttentionDGCL in the code)
   - AttentiveFP, a single attention based GNN that works directly on the atom/bond graph without fingerprints

   Both are evaluated on the same held out scaffold split so their RMSE, R2 and MAE numbers are directly comparable.

## Repository layout

```
1.compound_data_aquisation.ipynb
2.unwanted_filtering.ipynb
3.gnn.ipynb
encoder_gnn.py            GATNet and GINNet encoder definitions
model_gnn_fp_downstream.py  fusion model combining both encoders with fingerprints
pubchemfp.py               PubChem fingerprint calculation helper
data/
  CDK_compounds.csv
  CDK_compounds_lipinski.csv
  CDK_compounds_lipinski_NOPAINS.csv
  CDK_compounds_lipinski_NOPAINS_NOBRENKS.csv
```

## Results so far

On the scaffold split test set (about 500 held out molecules with structures not seen in training):

| Model | Test RMSE | Test R2 | Test MAE |
|---|---|---|---|
| AttentionDGCL (GAT + GIN + fingerprints) | 1.07 | 0.22-0.31 | 0.79 |
| AttentiveFP (graph only) | 1.06 | 0.23 | 0.80 |

Both models land in roughly the same range despite very different architectures, which points to the scaffold split and the relatively small training set (around 4000 molecules) being the main limiting factor rather than model choice. A pIC50 MAE of around 0.8 corresponds to roughly a 6-fold average error in predicted IC50, which is fine for ranking/triaging candidates but not for precise potency prediction.

## Requirements

- Python 3.11
- rdkit
- torch and torch_geometric
- deepchem (used for the scaffold splitting utilities and some helper functions)
- chembl_webresource_client
- pandas, numpy, scikit-learn, scipy, matplotlib
- tqdm

## Notes and known limitations

- The dataset is small by deep learning standards, so both GNN models show a noticeable gap between training and test performance. Ideas for improving this are pretraining the encoders on a larger unlabeled molecule set before fine-tuning on CDK2, adding SMILES based data augmentation, or running the scaffold split with several different seeds to get a steadier estimate of performance.
- A classical baseline (Random Forest or gradient boosting on the fingerprints alone) has not been added yet, which would be a useful sanity check against the GNN results.
- The `encoder_gnn.py` and `model_gnn_fp_downstream.py` files in this repo were reconstructed from earlier notebook outputs after the originals were lost, so their exact internals may not perfectly match whatever produced the very first training run.
# molgnn-cdk2-affinity
# molgnn-cdk2-affinity
