#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: creating 24th Mar. 2019

@Author: Ling
@Description: Faiss query

"""

import numpy as np
import faiss
import h5py
import time

from .cnn_feature import CNNFeature


class FaissQuery:
    """
    Giving query_feature_db and feature of query image, compute similarity
    """
    def __init__(self, query_hdf5, model_path, model_type='vgg16', k=1):
        self.query_hdf5 = query_hdf5

        self.d, self.xb, self.xq = None, None, None
        self.k = k
        self.index = None

        self.query_names = None

        self.get_data_and_train()

        self.model = CNNFeature(model_path, model_type)

    def get_data_and_train(self):
        """
        Load query_feature_db and train faiss index
        :return: 
        """
        # load query_feature_db
        query_h5f = h5py.File(self.query_hdf5, 'r')
        query_feats = query_h5f['feature'][:]
        self.query_names = query_h5f['name'][:]

        # set faiss parameters
        self.d = query_feats.shape[1]
        self.xb = query_feats

        # train index of faiss
        index = faiss.IndexFlatL2(self.d)
        index.add(self.xb)
        self.index = index

    def query(self, img_path):
        """
        query for computing similarity
        :param img_path: path of image query
        :return: Top1 similarity, Top1 image index
        """
        norm_feat = self.model.get_feature(img_path)
        self.xq = np.array([norm_feat])
        dis, idx = self.index.search(self.xq, self.k)

        img_save_path = self.query_names[idx[0]][0].decode('utf-8')

        return dis[0], img_save_path

    def query_test(self, img_path):
        """
        query for computing similarity
        :param img_path: path of image query
        :return: TopK similarity, TopK image index
        """
        norm_feat = self.model.get_feature(img_path)
        self.xq = np.array([norm_feat])
        dis, idx = self.index.search(self.xq, self.k)

        img_save_path = [i.decode('utf-8') for i in self.query_names[idx[0]]]

        return dis, img_save_path
