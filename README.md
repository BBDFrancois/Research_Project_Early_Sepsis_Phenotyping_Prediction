# Sepsis Early Prediction — MIMIC-III

Research project by François Zapletal (ESILV) under the supervision of Nédra Mellouli : hybrid and explainable architecture for early sepsis phenotyping and prediction using MIMIC-III data.
The pipeline combines deep learning-based representation learning with a mixture-of-experts classifier
and an attention-based gating network for patient phenotyping.

## Architecture

```
sepsis-prediction/
├── src/
│   ├── processor.py          # Data loading, windowing, filtering, train/test split
│   ├── embedder.py           # SAITS + TS2Vec + Autoencoder encoding pipeline
│   ├── gating_network.py     # Attention gating network (cluster routing)
│   ├── moe.py                # Mixture of Experts classifier and diagnostics
│   └── main.ipynb            # Main notebook (full end-to-end run)
├── data/
│   ├── Dataset.csv           # Raw MIMIC-III dataset (not versioned)
│   └── sepsis_processed_full.npz  # Pre-computed encoded representations (generated)
├── models/
│   ├── saits_model_base.pth
│   ├── ts2vec_model_base.pth
│   ├── autoencoder_model_base.pth
│   ├── saits_model_centered.pth
│   ├── ts2vec_model_centered.pth
│   └── autoencoder_model_centered.pth
├── annex/                    # Generated figures and outputs
├── .gitignore
└── README.md
```

## Pipeline Overview

1. **Preprocessing** (processor.py) : window extraction around sepsis onset, variable filtering, standardization
2. **Encoding** (embedder.py) : SAITS imputation of missing values, TS2Vec temporal embedding, autoencoder compression
3. **Phenotyping** : KMeans (k=4) clustering on peri-onset embeddings to define sepsis phenotypes
4. **Gating Network** (gating_network.py) : learns to predict future phenotype from pre-onset embedding
5. **Mixture of Experts** (moe.py) : routes patients to specialized MLP classifiers based on gating weights

## Requirements

torch
torch_directml
pypots
scikit-learn
pandas
numpy
matplotlib
seaborn
scipy

Reminded at the start of the notebook main


## Usage

Downlaod the dataset MIMIC III and rename it Dataset.csv et put it in the data folder
Then run downlaod the python librairies
After you can run all the pythons files in the src folder (src/processor.py src/embedder.py src/gating_network.py src/moe.py)
Then open src/main.ipynb and run cells sequentially.
If encoded data is already saved (data/sepsis_processed_full.npz), skip to section 3 (Load Pre-Computed Representations).

## Data

The raw dataset (data/Dataset.csv) is derived from MIMIC-III and is not included in this repository.
Access requires credentialed registration on PhysioNet
