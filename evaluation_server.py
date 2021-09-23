import pandas as pd
import numpy as np
import random
import pickle
from Loader_val import SignedPairsDataset, SinglePeptideDataset, get_index_dicts
import sklearn.metrics as metrics
from Trainer_2_mcpas import ERGOLightning
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from argparse import Namespace
from argparse import ArgumentParser
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score


def load_model(hparams, checkpoint_path):
    model = ERGOLightning(hparams)
    # checkpoint = torch.load(checkpoint_path, map_location='cuda:1')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    return model


def Merge(dict1, dict2):
    dict2.update(dict1)
    return ((dict2))


def evaluate(hparams, model, data):
    samples = []

    with open(hparams.dataset + '_train_samples.pickle', 'rb') as handle:
        train = pickle.load(handle)

    test_dataset = SignedPairsDataset(data, get_index_dicts(train))

    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=lambda b: test_dataset.collate(b, tcr_encoding=hparams.tcr_encoding_model,
                                                                  cat_encoding=hparams.cat_encoding))
    outputs = []
    dict_list = []
    for batch_idx, batch in enumerate(loader):
        outputs.append(model.validation_step(batch, batch_idx))
        temp_dict = Merge(outputs[-1], test_dataset.data[batch_idx])
        # temp_dict['y'] = int(temp_dict['y'].data)
        temp_dict['prediction'] = float(temp_dict['y_hat'].data)
        temp_dict['val_loss'] = float(temp_dict['val_loss'].data)
        dict_list.append(temp_dict)

    df = pd.DataFrame.from_dict(dict_list)
    df = df[['tcrb', 'vb', 'jb', 'tcra', 'va', 'ja', 't_cell_type', 'mhc', 'prediction']]
    return df


def run(file_pickle):
    with open(file_pickle, 'rb') as pick_file:
        data = pickle.load(pick_file)
    # data = pickle.load(file_pickle)
    checkpoint_path = "checkpoint/epoch=32.ckpt"

    args = {'dataset': 'final_mcpas_neg_mhc_2206', 'tcr_encoding_model': 'AE', 'cat_encoding': 'embedding',
            'use_alpha': True, 'use_vj': True, 'use_mhc': True, 'use_t_type': True, 'aa_embedding_dim': 10,
            'cat_embedding_dim': 50, 'embedding_dim': 10, 'lstm_dim': 500, 'weight_factor': 1,
            'lr': 0.001, 'dropout': 0.3, 'wd': 0.0005, 'encoding_dim': 100}

    hparams = Namespace(**args)
    checkpoint = checkpoint_path
    model = load_model(hparams, checkpoint)
    df = evaluate(hparams, model, data)
    return df


def main():
    df = run('final_mcpas_neg_mhc_2206_train_samples.pickle')
    df.to_csv('output.csv')


if __name__ == "__main__":
    main()
