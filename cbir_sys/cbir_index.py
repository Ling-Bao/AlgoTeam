# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Ling Bao

@Function: Indexing image
@Date: creating 24th Mar. 2019
"""

import argparse

from common.index_image import IndexImg


ap = argparse.ArgumentParser()
ap.add_argument("-model_path", required=True,
                help="model path for image's feature extract")
ap.add_argument("-save_path", required=True,
                help="Save path of index file")
ap.add_argument("-img_root", required=True,
                help="Root to images which contains images to be indexed")
args = vars(ap.parse_args())


if __name__ == '__main__':
    """ h5_model_path--path of model, h5_save_path--feature save path, img_root--root of images """
    h5_model_path = args["model_path"]
    h5_save_path = args["save_path"]
    img_root = args["img_root"]

    index_img = IndexImg()
    index_img.add_feature(h5_model_path, h5_save_path, img_root)
