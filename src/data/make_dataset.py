# -*- coding: utf-8 -*-
import logging
import pickle
from glob import glob
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train_images = []
    train_labels = []
    train_filepaths = glob(input_filepath+'/train*')
    for fp in train_filepaths:
        train_i = np.load(fp, allow_pickle=True)
        train_images.append(train_i['images'])
        train_labels.append(train_i['labels'])
    train_images = np.concatenate(train_images)
    train_images = (train_images-train_images.mean(axis=0))/(train_images.std(axis=0)+1e-9)
    train_labels = np.concatenate(train_labels)
    with open(output_filepath+'/corruptmnist_train.pkl', 'wb') as fp:
        pickle.dump((torch.from_numpy(train_images), torch.from_numpy(train_labels)), fp)
    test = np.load(input_filepath+"/test.npz", allow_pickle=True)
    test_images = (test['images']-train_images.mean(axis=0))/(train_images.std(axis=0)+1e-9)
    test_labels = test['labels']
    with open(output_filepath+'/corruptmnist_test.pkl', 'wb') as fp:
        pickle.dump((torch.from_numpy(test_images), torch.from_numpy(test_labels)), fp)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
