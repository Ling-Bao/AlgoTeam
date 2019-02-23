#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建于2019年1月14日

@作者: 包灵
@描述: faiss库使用

@可重用性：弱
"""

import numpy as np
import faiss
import h5py
import time

from .cnn_feature import CNNFeature


class FaissLearn:
    """
    计算先验库与采集库的距离
    """
    def __init__(self, db_hdf5, query_hdf5, k=1):
        self.db_hdf5 = db_hdf5
        self.query_hdf5 = query_hdf5
        self.k = k

        self.d, self.nb, self.nq = None, None, None
        self.xb, self.xq = None, None

        self.db_names, self.query_names = None, None

        self.get_data()

    def get_data(self):
        """
        载入先验库、采集库数据，并设置faiss参数
        :return: 
        """
        # 载入先验库
        db_h5f = h5py.File(self.db_hdf5, 'r')
        db_feats = db_h5f['feature'][:]
        self.db_names = db_h5f['name'][:]

        # 载入采集库
        query_h5f = h5py.File(self.query_hdf5, 'r')
        query_feats = query_h5f['feature'][:]
        self.query_names = query_h5f['name'][:]

        # 设置faiss参数
        self.d = db_feats.shape[1]
        self.nb = len(self.db_names)
        self.xb = db_feats
        self.nq = len(self.query_names)
        self.xq = query_feats

    def flat_l2_run(self, save_path, thresh=0.50):
        """
        计算采集库与采集库距离，并统计各阈值下能召回的采集库图像数量；最后保存召回的采集库图像形成查询库
        :param save_path: 查询库保存路径
        :param thresh: 判别阈值， eg. 0.5
        :return: None
        """
        index = faiss.IndexFlatL2(self.d)
        index.add(self.xb)

        t1 = time.time()
        dis, idx = index.search(self.xq, self.k)
        t2 = time.time()
        print('Time: %.5f' % (t2 - t1))

        print('Distance = %.5f' % np.min(dis[:, 0]))
        print('Recall = %d 张' % len(np.where(dis[:, 0] > thresh)[0]))

        keep_features = self.xq[np.where(dis[:, 0] > thresh)[0]]
        keep_names = self.query_names[np.where(dis[:, 0] > thresh)[0]]

        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('feature', data=keep_features)
        h5f.create_dataset('name', data=np.string_(keep_names.tolist()))
        h5f.close()


class FaissQuery:
    """
    给定查询库、图像特征表示模型条件下，实现查询图像与查询库距离的计算
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
        载入查询库，并完成faiss索引训练
        :return: 
        """
        # 载入检索库
        query_h5f = h5py.File(self.query_hdf5, 'r')
        query_feats = query_h5f['feature'][:]
        self.query_names = query_h5f['name'][:]

        # 设置faiss参数
        self.d = query_feats.shape[1]
        self.xb = query_feats

        # faiss训练
        index = faiss.IndexFlatL2(self.d)
        index.add(self.xb)
        self.index = index

    def query(self, img_path):
        """
        实现查询图像与查询库距离计算
        :param img_path: 查询图像路径
        :return: Top1图像与查询图像距离，Top1图像名
        """
        t1 = time.time()
        norm_feat = self.model.get_feature(img_path)
        self.xq = np.array([norm_feat])
        dis, idx = self.index.search(self.xq, self.k)
        t2 = time.time()

        # print('Time: %.5f' % (t2 - t1))

        img_save_path = self.query_names[idx[0]][0].decode('utf-8')

        return dis[0], img_save_path

    def query_test(self, img_path):
        """
        实现查询图像与检索库距离计算
        :param img_path: 查询图像路径
        :return: TopK图像与查询图像距离，TopK图像名
        """
        t1 = time.time()
        norm_feat = self.model.get_feature(img_path)
        self.xq = np.array([norm_feat])
        dis, idx = self.index.search(self.xq, self.k)
        t2 = time.time()

        # print('Time: %.5f' % (t2 - t1))

        img_save_path = [i.decode('utf-8') for i in self.query_names[idx[0]]]

        return dis, img_save_path


# if __name__ == '__main__':
#     """ 可用于特定场景识别 """
#     db_path = 'path_to/image_root/'
#     query_path = 'path_to/query.hdf5'
#
#     h5_keep_path = 'keep.hdf5'
#     faiss_learn = FaissLearn(db_path, query_path)
#     faiss_learn.flat_l2_run(h5_keep_path, thresh=0.5)
#
#     h5_model_path = 'path_to/vgg16.h5'
#     img_file = 'path_to/test.jpg'
#     faiss_query = FaissQuery(keep_save_path, h5_model_path, k=5)
#     distance, img_name = faiss_query.query(img_file)
#
#     distances, img_names = faiss_query.query_test(img_file)
#
#     print('Distance = %.5f, Image path = %s' % (distance, img_name))
