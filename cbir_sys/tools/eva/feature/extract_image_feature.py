# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Ling Bao

@Function: Indexing image
@Date: creating 6th Apr. 2019
"""

import os
import h5py
import numpy as np

from .cnn_feature import CNNFeature


class FeatureExtract:
    """
    @Description: 1. extract feature
                  2. save feature to hdf5 database
    """
    def __init__(self, im_list, q_im_list, save_path, model_path, model_type='vgg16'):
        self.model_path, self.model_type = model_path, model_type

        self.imlist, self.qimlist, self.save_path = im_list, q_im_list, save_path
        self.X, self.Q = None, None

    def run(self):
        """
        extract feature for evaluation
        :return: None
        """
        self.__get_feature()

        self.__save_hdf5()

    def __get_feature(self):
        """
        1. extract feature for images
        :return: None
        """
        # feature creator
        self.model = CNNFeature(self.model_path, self.model_type)

        # extract
        x_feature = []
        for i, img_path in enumerate(self.imlist):
            norm_feat = self.model.get_feature(img_path)

            x_feature.append(norm_feat)

            print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(self.imlist)))

        self.X = np.array(x_feature)

        q_feature = []
        for i, img_path in enumerate(self.qimlist):
            norm_feat = self.model.get_feature(img_path)

            q_feature.append(norm_feat)

            print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(self.qimlist)))

        self.Q = np.array(q_feature)

    def __save_hdf5(self):
        """
        2. save feature to hdf5 database
        :return: None
        """
        h5f = h5py.File(self.save_path, 'w')
        h5f.create_dataset('X', data=self.X)
        h5f.create_dataset('Q', data=self.Q)
        h5f.close()
