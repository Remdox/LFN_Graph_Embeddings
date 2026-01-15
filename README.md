# LFN Graph Embeddings (project APEROL)
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
[Read the first proposal](./reports/first_proposal/first_proposal.pdf).  
[Read the midterm report](./reports/midterm_report.pdf).  
[Read the final report](./reports/final_report.pdf).
   
# Instructions
## Requirements
### Dependencies
See [the dedicated file](./requirements.txt) and install the dependencies inside a virtual environment.
Additionaly, the package torch-cluster needs to be installed in order to use Node2Vec:
```
pip install torch-cluster
```

The program has been tested on GPU using these additional packages with specific versions:
* Pytorch 2.8.0, CUDA 1.28
* cupy-cuda12x
* torch-cluster needs to be reinstalled with its gpu-adiacent version:
   ```
   pip install torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
   ```
### Hardware
The program runs on a single NVIDIA GPU, if detected; otherwise, it can run on cpu with good performance for small and medium-sized datasets.

For the few who have access to the cluster "Blade", a [.def file](./LFN_container_gpu.def) is provided so that you can build the container starting from an already existing image cv-ml-torch.sif obtainable from inside the cluster. Before building, please change the path of the localimage in the .def file.

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
   You can also choose to run the pipeline with a single dataset, as well as a single embedding algorithm and a single downstream model. One example:
   ```python
   python embeddings_pipeline.py --data datasets/processed_datasets/Bio_grid_fission_yeast.csv
   ```

   Or even:
   ```python
   python embeddings_pipeline.py --data Bio_grid_fission_yeast --embed DVNE --model MLP
   ```
   
# Datasets and project structure
9 datasets of different sizes are used, ranging from ~25k edges to ~3M edges. You can check the references for each dataset in the [midterm report](./reports/midterm_report.pdf).

## Datasets full details

| Network | Nodes | Edges | Type | Link to the dataset page
| :--- | :--- | :--- | :--- | :--- | 
| **Pennsylvania** | 1,088,092 | 3,083,796 | Directed | http://snap.stanford.edu/data/roadNet-PA.html |
| **Padua** (province) | 122,680 | 304,184 | Directed | https://github.com/Remdox/Padua_Network_dataset_2025 |
| **Hong Kong** (city) | 43,620 | 91,542 | Directed | https://github.com/yzengal/RoadNetwork-China-City/blob/main/Hongkong.road-d.tar.gz |
| **Italian Covid-19 Retweet Network** | 221,574 | 800,000 | Directed | https://zenodo.org/records/13909011 |
| **Deezer** | 143,884 | 846,915 | Undirected | https://snap.stanford.edu/data/gemsec-Deezer.html |
| **GitHub Developers** | 37,700 | 289,003 | Undirected | http://snap.stanford.edu/data/github-social.html |
| **Mus Musculus** Protein Interactions (confidence score >0.7, only AB) | 20,969 | 800,000 | Undirected | https://string-db.org/cgi/download?sessionId=b9zuGHnAZu39&species_text=Mus+musculus&settings_expanded=1&min_download_score=400&filter_redundant_pairs=1&delimiter_type=csv |
| **Saccharomyces cerevisiae** Protein Interactions (confidence score >0.4, only AB) | 5,786 | 100,000 | Undirected | https://string-db.org/cgi/download?sessionId=b9zuGHnAZu39&species_text=Saccharomyces+cerevisiae&settings_expanded=1&min_download_score=700&filter_redundant_pairs=1&delimiter_type=csv |
| **Bio-grid-fission-yeast** | 2,000 | 25,300 | Undirected | https://networkrepository.com/bio-grid-fission-yeast.php |
---
## Project Structure
The project structure is defined as:
```
LFN_Graph_Embeddings/
├── datasets/
│   └── datasets_info.csv
│   └── original_datasets.zip
└── include/
│   └── graphsage/
│   └── line/
│   └── node2vec/
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
│   └── dataset_utils.py
│   └── embeddings.py
│   └── models.py
│   └── pipeline.py
│   └── pipeline_utils.py
│   └── utils.py
└── README-md
```
Where The [datasets_info.csv](./datasets/datasets_info.csv) file provides the fields considered for each of the datasets used. If a new dataset is included, this file has to be correctly updated before running the program.

# Results
See the [final report](./reports/final_report/final_report.pdf).

# Credits
We thank the creators of the following implementations (see the [include folder](./include/) inside the project):
* [Node2Vec](https://github.com/palash1992/GEM);
* [LINE](https://github.com/tangjianpku/LINE);
* [Deep Variational Network Embedding in Wasserstein Space](https://github.com/Lakshya-99/Deep_Variational_Network_Embedding/tree/master);
* [GraphSage](https://github.com/williamleif/GraphSAGE).
