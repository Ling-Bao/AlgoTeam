# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Ling Bao

@Function: Indexing image
@Date: creating 24th Mar. 2019
"""

import os
import h5py
import numpy as np

from .cnn_feature import CNNFeature


class IndexImg:
    """
    @Description: 1. get all image path in path root
                  2. extract all images' feature
                  3. create index and save to hdf5 feature database
                  4. support increasing add feature to old hdf5 database
    """
    def __init__(self):
        self.model, self.model_path, self.model_type = None, None, None

        self.save_path, self.img_list = None, None
        self.features, self.names = None, None

    def add_feature(self, model_path, save_path, img_dir, model_type='vgg16', img_suffix=['.jpg', '.jpeg']):
        """
        
        :param model_path: load path for model
        :param save_path: path for feature save, using hdf5 format to save
        :param img_dir: root directory of image
        :param model_type: type os model, eg. vgg16
        :param img_suffix: suffix of image, eg. ['.jpg', '.jpeg']
        :return: None
        """
        self.model_path = model_path
        self.model_type = model_type
        self.save_path = save_path

        self.__get_image_path(img_dir, img_suffix)

        self.__get_feature()

        self.__save_hdf5()

    def __get_image_path(self, img_dir, img_suffix):
        """
        1. get all image path in image directory
        :param img_dir: root directory of image
        :param img_suffix: suffix of image, eg. ['.jpg', '.jpeg']
        :return: 
        """
        img_list = []
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if os.path.splitext(file)[1] in img_suffix:
                    img_list.append(os.path.join(root, file).encode('utf-8'))

        self.img_list = img_list

        return img_list

    def __get_feature(self):
        """
        2. extract feature
        :return: features, names
        """
        self.model = CNNFeature(self.model_path, self.model_type)

        feats = []
        names = []
        for i, img_path in enumerate(self.img_list):
            norm_feat = self.model.get_feature(img_path)

            feats.append(norm_feat)
            names.append(img_path)

            print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(self.img_list)))

        self.features = np.array(feats)
        self.names = names

        return feats, names

    def __save_hdf5(self):
        """
        3. create index and save to hdf5 feature database
        4. support increasing add feature to old hdf5 database
        :return: 
        """
        # check whether exists feature extracted
        if os.path.exists(self.save_path):
            db_h5f = h5py.File(self.save_path, 'r')
            db_feats = db_h5f['feature'][:]
            db_img_names = db_h5f['name'][:]

            self.features = np.append(self.features, db_feats, 0)
            self.names = self.names + db_img_names.tolist()
            db_h5f.close()

        h5f = h5py.File(self.save_path, 'w')
        h5f.create_dataset('feature', data=self.features)
        h5f.create_dataset('name', data=np.string_(self.names))
        h5f.close()
