# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Ling Bao

@Function: Indexing image
@date: 23/2/2019
"""

import argparse

from cbir.index_image import IndexImg


ap = argparse.ArgumentParser()
ap.add_argument("-model_path", required=True,
                help="model path for image's feature extract")
ap.add_argument("-save_path", required=True,
                help="Save path of index file")
ap.add_argument("-img_root", required=True,
                help="Root to images which contains images to be indexed")
args = vars(ap.parse_args())


if __name__ == '__main__':
    """ 实现图像特征表示提取：model_path--特征提取模型路径，save_path--特征提取库保存路径，database--图像集路径 """
    h5_model_path = args["model_path"]
    h5_save_path = args["save_path"]
    img_root = args["img_root"]

    index_img = IndexImg()
    index_img.add_feature(h5_model_path, h5_save_path, img_root)
