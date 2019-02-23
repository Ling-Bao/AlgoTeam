# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Ling Bao

@Function: Get image's representation using pre-trained model
@date: 28/1/2019
"""

import numpy as np
from numpy import linalg as LA

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.models import load_model


class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'

        self.model = None

    def get_trained_model(self, model_path):
        """
        下载VGG16预训练模型到~/.keras/models；创建VGG16 model实例实现特征提取；保存模型与权重到model_path
        :param model_path: 模型保存路径
        :return: None
        """
        self.model = VGG16(weights=self.weight,
                           input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                           pooling=self.pooling, include_top=False)

        # test
        self.model.predict(np.zeros((1, 224, 224, 3)))

        # save model and weights
        self.model.save(model_path)

    def get_product_mode(self, model_path):
        """
        从model_path载入VGG16模型并创建VGG16 model实例实现特征提取
        :param model_path: 载入模型路径
        :return: None
        """
        # load model and weights
        self.model = load_model(model_path)

        # test
        self.model.predict(np.zeros((1, 224, 224, 3)))

    def extract_feat(self, img_path):
        """
        Use vgg16 model to extract features, Output normalized feature vector
        :param img_path: 待提取特征的图像路径
        :return: None
        """
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])

        return norm_feat


class CNNFeature:
    """
    CNN特征提取产品类
    """
    def __init__(self, model_path, model_type):
        self.model = None

        if model_type == 'vgg16':
            self.vgg16_feature(model_path)

    def vgg16_feature(self, model_path):
        self.model = VGGNet()
        self.model.get_product_mode(model_path)

    def get_feature(self, img_path):
        return self.model.extract_feat(img_path)
