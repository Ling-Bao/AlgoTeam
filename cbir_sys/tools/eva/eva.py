#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@function: Evaluation
@input: datasets
@output: mAP and mP@k

@Author: Ling Bao
@Date: creating 6th Apr. 2019
"""
import os

from feature.extract_image_feature import FeatureExtract
from python.my_evaluate import get_imlist, run_eva


def evaluation(data_root, feature_path, model_path, test_dataset='roxford5k', b_feature_extract=False):
    imlist, qimlist = get_imlist(data_root, test_dataset=test_dataset)

    if b_feature_extract:
        feature_extract = FeatureExtract(imlist, qimlist, feature_path, model_path)
        feature_extract.run()

    run_eva(feature_path, data_root, test_dataset=test_dataset)


if __name__ == '__main__':
    dataset_root = '/home/bl/workspace/datas/cbir/data'

    # Set test dataset: roxford5k | rparis6k
    test_dataset_name = 'rparis6k'
    feature_file = os.path.join(dataset_root, test_dataset_name + '_features.hdf5')

    model_file = '../../support/vgg16.h5'

    evaluation(dataset_root, feature_file, model_file, test_dataset=test_dataset_name, b_feature_extract=False)
