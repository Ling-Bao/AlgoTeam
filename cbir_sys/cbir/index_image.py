# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Ling Bao

@Function: Indexing image
@date: 28/1/2019
"""

import os
import h5py
import numpy as np

from .cnn_feature import CNNFeature


class IndexImg:
    """
    @Description: 1. 获取指定目录下图像路径；2. 提取图像特征；3. 以hdf5格式按索引保存图像特征；4. 支持增量添加图像特征。
    """
    def __init__(self):
        self.model, self.model_path, self.model_type = None, None, None

        self.save_path, self.img_list = None, None
        self.features, self.names = None, None

    def add_feature(self, model_path, save_path, img_dir, model_type='vgg16', img_suffix=['.jpg', '.jpeg']):
        """
        
        :param model_path: keras模型路径
        :param save_path: 特征保存路径，以hdf5格式保存
        :param img_dir: 图像路径，可包含多层文件夹
        :param model_type: 模型类型， eg. vgg16
        :param img_suffix: 图像后缀名list， eg. ['.jpg', '.jpeg']
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
        1. 获取指定目录下图像路径
        :param img_dir: 图像路径，可包含多层文件夹
        :param img_suffix: 图像后缀名list, eg. ['.jpg', '.jpeg']
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
        2. 提取图像特征，返回图像特征及图像对应路径名
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
        3. 以hdf5格式按索引保存图像特征；
        4. 支持增量添加图像特征。
        :return: 
        """
        # 检查是否存在已经保存的图像特征
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
