# LFN Graph Embeddings
LFN project, 2025.

Table of Contents
=================

   * [Introduction](#introduction)
   * [Instructions](#instructions)
     * [Requirements](#requirements)
     * [Running the project](#running-the-project)
   * [Datasets and project structure](#datasets-and-project-structure)
      * [Project structure](#project-structure)
   * [Code](#code)
      * [1.Datasets preprocessing](#1datasets-preprocessing)
      * [2.Setting up the link prediction task](#2setting-up-the-link-prediction-task)
      * [3.Extracting node embeddings from the data](#3extracting-node-embeddings-from-the-data)
      * [4.Prediction using the embeddings](#4prediction-using-the-embeddings)
      * [5.Metrics and Experiments](#5metrics-and-experiments)
      * [6.Output](#6output)
   * [Results](#results)

# Introduction
[Read the full proposal](./reports/first_proposal/first_proposal.pdf).  
[Read the midterm report](./reports/first_proposal.pdf).
   
# Instructions
## Requirements
For the embeddings, we use the following implementations:
* [Node2Vec](https://github.com/palash1992/GEM);
* [LINE](https://github.com/tangjianpku/LINE);
* [Deep Variational Network Embedding in Wasserstein Space](https://github.com/Lakshya-99/Deep_Variational_Network_Embedding/tree/master);
* [GraphSage](https://github.com/williamleif/GraphSAGE).

You can run the [setup file](./setup.py) in a venv environment for automatic installation.
**HARDWARE**

## Running the project
* Extract the archive [./datasets/original_dataset.zip](original_dataset.zip) containing the datasets we used.
* Run [the preprocessing of the datasets](./src/dataset_preprocessing.py):
    ```python
    python dataset_preprocessing.py
    ```
* Run [the program](./src/embeddings_pipeline.py):
    ```python
    python embeddings_pipeline.py
    ```
   
# Datasets and project structure
9 datasets of different sizes are used, ranging from ~25k edges to ~3M edges. You can check the fulll details and references for each dataset in the [proposal](./reports/first_proposal/first_proposal.pdf).

## Project Structure
The dataset structure is defined as:
```
LFN_Graph_Embeddings/
├── datasets/
│   └── datasets_info.csv
│   └── original_datasets.zip
└── reports/
│   └── final_report/
│   |   └── final_report.pdf
│   |   └── final_report.tex
│   │   └──   ...
│   └── first_proposal/
│   |   └── first_proposal.pdf
│   |   └── first_proposal.tex
│   │   └──   ...
|   └── midterm_report/
│   |   └── midterm_report.pdf
│   |   └── midterm_report.tex
│   │   └──   ...
└── src/
│   └── datasets_preprocessing.py
│   └── emebeddings_pipeline.py
└── README-md
```
Where The [datasets_info.csv](./datasets/datasets_info.csv) file provides the fields considered for each of the datasets used.

# Code
The pipeline is:
## 1.Datasets preprocessing
## 2.Setting up the link prediction task
## 3.Extracting node embeddings from the data
## 4.Prediction using the embeddings
## 5.Metrics and Experiments
## 6.Output

# Results
See the [final report](./reports/final_report/final_report.pdf).
