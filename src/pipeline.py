import pandas as pd
import numpy  as np
import random
import torch
import time
import click
import copy
from pathlib import Path
import dataset_utils
from dataset_utils import Graph
import pipeline_utils
from pipeline_utils import sample_negative_edges, split_graph_data, merge_negative_edges, embed_edges
import embeddings
from embeddings import GraphSage, Node2Vec, LINE
import models
from models import SVM, MLP, RandomForest
import model_utils
from model_utils import evaluate_AUROC, evaluate_AUPR

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
    MLP_num_epochs = 200
    MLP_patience = 20

    datasets = dataset_utils.get_datasets()
    if data:
        dataset_name = Path(data).stem
        dataset = datasets.pop(dataset_name, None)
        if dataset is None:
            raise ValueError("The specified dataset is not available")
        datasets.clear()
        datasets[dataset_name] = dataset # my_dataset Bio_grid_fission_yeast

    embed_methods = {"GraphSage":GraphSage, "Node2Vec":Node2Vec, "Line":LINE}
    if embed:
        embed_methods = {embed: embed_methods[embed]}
    for name, cls in embed_methods.items():
        embed_methods[name] = cls()

    models = {"SVM":SVM, "MLP":MLP, "RandomForest":RandomForest}
    if model:
        models = {model: models[model]}
    for name, cls in models.items():
        models[name] = cls()

    for data_name, dataset in datasets.items():
        print(f"@@@ DATASET: {data_name} @@@")
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

        if embed_methods.get('DVNE') is not None:
            print(embed_methods['DVNE'].sample_triplets(G_embed, G_pred.graph_data.edge_index))

        for name, emb_method in embed_methods.items():
            print(f"Training: {name}")
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
            print(f"---- Embedding edges with method: {method_name} ----")
            embedded_train = embed_edges(G_train, method)
            embedded_val   = embed_edges(G_val, method)
            embedded_test  = embed_edges(G_test, method)

            print("Embeddings done")

            # train with the embedding of the training set

            for name, mod in models.items():
                print(f"---- Train, validation, test of model {name} ----")
                print(f"Training model {name}")
                if name == "MLP":
                    best_val_loss = float('inf')
                    epochs_wout_improvement = 0
                    best_model_wts = copy.deepcopy(mod.state_dict())
                    for epoch in range(1, MLP_num_epochs):
                        train_loss = mod.train_model(embedded_train, train_labels)

                        mod.eval()
                        with torch.no_grad():
                            val_loss = mod.criterion(mod(embedded_val), val_labels.long()).item()

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_wts = copy.deepcopy(mod.state_dict())
                            epochs_wout_improvement = 0
                        else:
                            epochs_wout_improvement += 1

                        if epochs_wout_improvement == MLP_patience:
                            print(f"Early Stopping at {epoch} epochs")
                            break

                        if epoch % 50 == 0:
                            print(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
                    mod.load_state_dict(best_model_wts)
                else:
                    mod.train_model(embedded_train, train_labels)

                print(f"Testing model {name}")
                pred = mod.predict(embedded_test)

                # metrics
                print(f"AUROC: {evaluate_AUROC(test_labels, pred)}")
                print(f"AUPR: {evaluate_AUPR(test_labels, pred)}\n")
                 # GS+MLP
                 # 0.8677982550730319 AUROC
                 # 0.7467608911257131 AURPR

                 # N2V+MLP
                 # 0.8974720615625918 AUROC
                 # 0.8209333103906062 AURPR


if __name__ == "__main__":
    main()
