import pandas as pd
import numpy  as np
import random
import torch
import time
import click
from pathlib import Path
import dataset_utils
from dataset_utils import Graph
import pipeline_utils
from pipeline_utils import sample_negative_edges
from pipeline_utils import split_graph_data
from pipeline_utils import merge_negative_edges
from pipeline_utils import embed_edges
import embeddings
from embeddings import GraphSage
from embeddings import Node2Vec
import models
import model_utils
from model_utils import evaluate_AUROC, evaluate_AUPR
from models import SVM, MLP

RANDOM_SEED = 104

E_EMBED_RATIO = 0.8
E_PRED_RATIO  = 1 - E_EMBED_RATIO

E_TRAIN_RATIO = 0.6
E_TEST_RATIO  = 0.2
E_VAL_RATIO   = 1 - E_TRAIN_RATIO - E_TEST_RATIO

@click.command()
@click.option("--data", required=False, help="Choose the name of a single dataset to analyze")
@click.option("--embed", required=False, help="Choose the name of a single embedding method to use")
@click.option("--model", required=False, help="Choose the name of a single classification model to use")
def main(data, embed, model):
    torch.manual_seed(RANDOM_SEED)
    random.seed(a=RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    negative_sample_ratio = 1 * E_PRED_RATIO

    datasets = dataset_utils.get_datasets()
    if data:
        dataset_name = Path(data).stem
        dataset = datasets.pop(dataset_name, None)
        if dataset is None:
            raise ValueError("The specified dataset is not available")
        datasets.clear()
        datasets[dataset_name] = dataset # my_dataset Bio_grid_fission_yeast

    embed_methods = {"GraphSage":GraphSage, "Node2Vec":Node2Vec}
    if embed:
        embed_methods = {embed: embed_methods[embed]}
    for name, cls in embed_methods.items():
        embed_methods[name] = cls()

    models = {"SVM":SVM, "MLP":MLP}
    if model:
        models = {model: models[model]}
    for name, cls in models.items():
        models[name] = cls()

    for name, dataset in datasets.items():
        print(f"@@@ DATASET: {name} @@@")
        # negative sampling: from E (edge set of whole graph) the size of E_pred negative edges
        print("---- negative sample ----")
        negative_sample_size = int(dataset.graph_data.num_edges * negative_sample_ratio)
        negative_sample = sample_negative_edges(dataset, negative_sample_size)
        print(negative_sample.graph_data)

        # embed split
        print("---- embed split ----")
        G_embed, _, G_pred = split_graph_data(dataset, val_ratio=0, test_ratio=0.2)
        print(G_embed.graph_data, _.graph_data, G_pred.graph_data)

        # TRAINING EMBEDDINGS

        for emb_method in embed_methods.values():
            emb_method.train_embed(G_embed)

        # train, test, validation split
        G_train, G_val, G_test = split_graph_data(G_pred, val_ratio=0.2, test_ratio=0.2)
        print("---- train, val, test split ----")
        print(G_train.graph_data, G_val.graph_data, G_test.graph_data)

        # add negative edges to G_train, G_val, G_test
        # the training set must have negative edges different from the other splits
        print("---- negative split ----")
        train_neg, val_neg, test_neg = split_graph_data(negative_sample, val_ratio=0.2, test_ratio=0.2)
        print(train_neg, val_neg, test_neg)
        print("---- negative merging ----")
        G_train, train_labels = merge_negative_edges(G_train, train_neg)
        G_val,   val_labels   = merge_negative_edges(G_val, val_neg)
        G_test,  test_labels  = merge_negative_edges(G_test, test_neg)
        print(G_train.graph_data, G_val.graph_data, G_test.graph_data)
        print("---- labels ----")
        print(train_labels, val_labels, test_labels)

        # GET EMBEDDINGS of all 3 classification datasets
        if embed_methods.get('GraphSage') is not None:
            embed_methods['GraphSage'].update_features(embed_methods['GraphSage'].compute_features(dataset))
            embed_methods['GraphSage'].update_adjacency(G_pred.graph_data.edge_index)


        for method_name, method in embed_methods.items():
            print(f"Embedding edges with method: {method_name}")
            print(G_train.graph_data.edge_index.shape)
            print(G_train.graph_data.edge_attr.shape)
            embedded_train = embed_edges(G_train, method)
            embedded_val   = embed_edges(G_val, method)
            embedded_test  = embed_edges(G_test, method)

            print("Embeddings done")

            # train with the embedding of the training set

            for name, mod in models.items():
                print(f"---- Train, validation, test of model {name} ----")
                print(f"Training model {name}")
                print(embedded_train.shape)
                mod.train_model(embedded_train, train_labels)

                # validate AND test ML classification models
                print(f"Validating model {name}")

                print(f"Testing model {name}")
                pred = mod.predict(embedded_test)

                # metrics
                print(f"AUROC: {evaluate_AUROC(test_labels, pred)}")
                print(f"AUPR: {evaluate_AUPR(test_labels, pred)}")


if __name__ == "__main__":
    main()
