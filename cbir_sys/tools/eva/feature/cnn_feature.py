# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Ling

@Function: Get image's representation using pre-trained model
@Date: creating 24th Mar. 2019
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
        Download pretrain model, eg. vgg16, to ~/.keras/models;
        Save model network and weights to model_path
        :param model_path: save path for model
        :return: None
        """
        self.model = VGG16(weights=self.weight,
                           input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                           pooling=self.pooling, include_top=False)

        # testing
        self.model.predict(np.zeros((1, self.input_shape[0], self.input_shape[1], self.input_shape[2])))

        # save model and weights to model_path
        self.model.save(model_path)

    def get_product_mode(self, model_path):
        """
        Using pretrain model, eg. vgg16, in model_path to create object and to extract feature
        :param model_path: load path for model
        :return: None
        """
        # load model and weights
        self.model = load_model(model_path)

        # testing
        self.model.predict(np.zeros((1, 224, 224, 3)))

    def extract_feat(self, img_path):
        """
        Use vgg16 model to extract features, Output normalized feature vector
        :param img_path: Path of images extracted feature
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
    Production of CNN feature extract
    """
    def __init__(self, model_path, model_type='vgg16'):
        self.model = None

        if model_type == 'vgg16':
            self.vgg16_feature(model_path)

    def vgg16_feature(self, model_path):
        self.model = VGGNet()
        self.model.get_product_mode(model_path)

    def get_feature(self, img_path):
        return self.model.extract_feat(img_path)
