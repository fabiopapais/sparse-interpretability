# Sparse Interpretability

Project Developed in the Large Language Models course at CIn/UFPE

## Overview
This project is a replication of the methods proposed in the paper "Sparse Autoencoders Find Highly Interpretable Features in Language Models". The core goal of the research is to address the challenge of polysemanticity in neural networks, a phenomenon where a single neuron activates in multiple, semantically distinct contexts, making models difficult to understand. 

The paper hypothesizes that polysemanticity is a result of superposition, where models represent more features than they have neurons by assigning them to directions in activation space. To resolve this, the authors propose training a sparse autoencoder on the internal activations of a language model. This autoencoder learns to represent the model's activations as a sparse combination of "features" from a learned dictionary. These features are shown to be more monosemantic and interpretable than those identified by other methods, allowing for a clearer understanding of the model's internal mechanisms. 

## Project Goal
The primary objective of this repository is to implement the sparse autoencoder architecture and replicate the key experiments and findings presented in the original paper. This includes training the autoencoder on language model activations and evaluating the interpretability of the learned features.

## Project Structure

<pre>
sparse-interpretability
│
├── README.md
├── requirements.txt
├── script.py
├── generate_activation_dataset.ipynb
│
└── edited_sparse_coding_files
    │
    ├── big_sweep.py
    ├── basic_l1_sweep.py
    │
    └── autoencoders
        │
        └── sae_ensemble.py
</pre>

* **README.md** → Project description and reference to the academic paper.
* **requirements.txt** → Python dependencies required to run the project.
* **script.py** → Script to extract model activations and save them to files.
* **generate_activation_dataset.ipynb** → Notebook to generate and visualize the activations dataset.
* **edited_sparse_coding_files/** → Directory with custom scripts for sparse coding and autoencoders.
    * **big_sweep.py** → Hyperparameter sweep experiments.
    * **basic_l1_sweep.py** → Basic sweep varying the L1 parameter.
    * **autoencoders/** → Autoencoder implementations.
        * **sae_ensemble.py** → Implementation of functional ensemble-type autoencoders.

## Original Research
All work is based on the following paper:

Title: SPARSE AUTOENCODERS FIND HIGHLY INTERPRETABLE FEATURES IN LANGUAGE MODELS 

Authors: Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, Lee Sharkey 

Publication: arXiv:2309.08600v3 [cs.LG] 4 Oct 2023 

Code: The original implementation can be found at https://github.com/HoagyC/sparse_coding 
