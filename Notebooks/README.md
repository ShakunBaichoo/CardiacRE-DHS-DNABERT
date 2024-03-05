<h1 align="center"> Heart-Specific Regulatory Elements Analysis with DNABERT</h1>
This folder contains Jupyter notebooks dedicated to the analysis of heart-specific regulatory elements leveraging the capabilities of DNABERT. The notebooks guide you through the process of identifying potential regulatory sequences in heart tissue, focusing on understanding their unique roles and functionalities. All files that are downloaded and processed are stored within this folder itself.

## Overview
There are two sets of Notebooks: one for Problem 1 and one for Problem 2.

### Problem 1 Notebooks
For Problem 1, the first two notebooks (focused on data preparation) are adapted versions of the notebooks from the [DNA Diffusion project](https://github.com/pinellolab/DNA-Diffusion/blob/main/notebooks/filter_master.ipynb), which are used to load the DHSs. 

- **Notebook 1**: Used to create the master dataset.
- **Notebook 2**: Used to filter out DHSs related to the heart organ. After filtering, the data is fed to a fine-tuned, pre-trained DNABERT model with `kmers=6`. The steps and details for running this step are provided in **DNABERT Prediction**. The analysis of the results is performed in the third notebook.

### Problem 2 Notebooks
For Problem 2, I have created my own version of the dataset using the DHSs and the vocabulary, then proceeded with DNABERT training and analysis of results, similar to Problem 1.

