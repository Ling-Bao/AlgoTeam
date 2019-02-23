#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建于2019年1月28日

@作者: 包灵
@描述: 合并两个hdf5检索库

@可重用性：中等
"""

import h5py
import numpy as np


def combine_hdf5(db_1, db_2, save_db):
    db_h5f = h5py.File(db_1, 'r')
    db_feats_1 = db_h5f['feature'][:]
    db_img_names_1 = db_h5f['name'][:]
    db_h5f.close()

    db_h5f = h5py.File(db_2, 'r')
    db_feats_2 = db_h5f['feature'][:]
    db_img_names_2 = db_h5f['name'][:]
    db_h5f.close()

    features = np.append(db_feats_1, db_feats_2, 0)
    names = db_img_names_1.tolist() + db_img_names_2.tolist()

    h5f = h5py.File(save_db, 'w')
    h5f.create_dataset('feature', data=features)
    h5f.create_dataset('name', data=np.string_(names))
    h5f.close()

    print('Query database is %d images' % features.shape[0])


if __name__ == '__main__':
    """ 合并两个检索库 """
    db_1_path = 'path_to/db1.hdf5'
    db_2_path = 'path_to/db2.hdf5'

    db_save_path = 'path_to/db.hdf5'
    combine_hdf5(db_1_path, db_2_path, db_save_path)
