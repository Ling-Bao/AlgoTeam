#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: creating 24th Mar. 2019

@author: ling
@description: generate triplet training data
"""

import faiss
import h5py
import os
import shutil
import json
import random
import numpy as np


def faiss_query(query_db_path, queried_db_path):
    """
    brute compute similary
    :param query_db_path:
    :param queried_db_path:
    :return: Distance, Index
    """
    # load feature db
    query_h5f = h5py.File(query_db_path, 'r')
    query_feats = query_h5f['feature'][:]
    query_names = query_h5f['name'][:]
    query_h5f.close()

    queried_h5f = h5py.File(queried_db_path, 'r')
    queried_feats = queried_h5f['feature'][:]
    queried_names = queried_h5f['name'][:]
    queried_h5f.close()

    # set faiss parameters
    d = queried_feats.shape[1]
    xb = queried_feats

    # train index of faiss
    index = faiss.IndexFlatL2(d)
    index.add(xb)

    # faiss query
    k = len(queried_feats)
    xq = query_feats
    dis, idx = index.search(xq, k)

    return dis, idx, query_names, queried_names


def get_full_path(img_path):
    root_image = 'pairs/'

    split_path_list = img_path.decode('utf-8').split('/')
    split_relative = split_path_list[
                     int(np.where(np.array(split_path_list) == 'open_source')[0][0] + 1):len(
                         split_path_list)]
    relative_path = '/'.join(split_relative)

    original_image_root = os.path.join(root_image, relative_path)

    if not os.path.exists(original_image_root):
        print('%s not exist' % img_path.decode('utf-8'))
        return None
    else:
        return original_image_root


def gen_triplet_data(db_root, save_path):
    """
    generate triplet datasets
    :param db_root: 特征库存放根目录
    :param save_path: 训练图像以及索引文件保存目录
    :return: 
    """
    db_list = [['xy_database.hdf5', 'query_database.hdf5'],
               ['xl_query.hdf5', 'xy_database.hdf5'],
               ['sw_query.hdf5', 'xy_database.hdf5'],
               ['cf_query.hdf5', 'xy_database.hdf5']]
    type_list = ['行李区', '室外', '出发厅']

    image_save_path = os.path.join(save_path, "images/")

    triplet_index = []
    group_id = 100000
    not_meet_num = 0
    all_num = 0
    for db_pair in db_list:
        q_db_path = os.path.join(db_root, db_pair[0])
        n_db_path = os.path.join(db_root, db_pair[1])

        p_dis, p_idx, p_query_names, p_queried_names = faiss_query(q_db_path, q_db_path)
        n_dis, n_idx, n_query_names, n_queried_names = faiss_query(q_db_path, n_db_path)

        all_num += len(p_queried_names)

        for i, img_path in enumerate(p_query_names):
            original_image_root = get_full_path(img_path)

            if original_image_root is None:
                continue

            img_name = img_path.decode('utf-8').split('/')[-1].split('.')[0]

            # 组id
            group_id += i

            # query
            query = str(group_id) + '_' + img_name + '_q.jpg'
            q_distance = 0.0
            q_type = type_scene

            # positive
            p_id_list = np.where(p_dis[i] < 0.35)[0][1:]
            if len(p_id_list):
                p_id = p_id_list[random.randint(0, len(p_id_list) - 1)]
                positive_full_path = get_full_path(p_queried_names[p_idx[i][p_id]])
                if positive_full_path is None:
                    continue

                positive_name = positive_full_path.split('/')[-1].split('.')[0]

                positive = str(group_id) + '_' + positive_name + '_p.jpg'
                p_distance = float(p_dis[i][p_id])
                p_type = type_scene
            else:
                not_meet_num += 1
                print('Positive Not meeting the requirements!')
                continue

            # negative
            n_id_list = np.where(n_dis[i] > 0.65)[0]
            if len(n_id_list):
                n_id = n_id_list[random.randint(0, len(n_id_list) - 1)]
                negative_full_path = get_full_path(n_queried_names[n_idx[i][n_id]])
                if negative_full_path is None:
                    continue

                negative_name = negative_full_path.split('/')[-1].split('.')[0]
                type_scene = negative_full_path.split('/')[-3]
                if type_scene in type_list:
                    pass
                else:
                    type_scene = 'negative'

                negative = str(group_id) + '_' + negative_name + '_n.jpg'
                n_distance = float(n_dis[i][n_id])
                n_type = type_scene
            else:
                not_meet_num += 1
                print('Negative Not meeting the requirements!')
                continue

            item_map = {}
            item_map['group_id'] = str(group_id)

            item = {}
            item['query'] = [query, q_distance, q_type]
            item['positive'] = [positive, p_distance, p_type]
            item['negative'] = [negative, n_distance, n_type]
            item_map['triplet'] = item

            triplet_index.append(item_map)

            # copy triplet images
            shutil.copy(original_image_root, os.path.join(image_save_path, query))
            shutil.copy(positive_full_path, os.path.join(image_save_path, positive))
            shutil.copy(negative_full_path, os.path.join(image_save_path, negative))

    fp = open(os.path.join(save_path, 'index.json'), 'w')
    json.dump(triplet_index, fp)

    print('All images: %d, Not meeting numbers: %d' % (all_num, not_meet_num))


if __name__ == '__main__':
    triplet_db_path = 'image_db/triplet_db/hdf5/'
    triplet_data_save = 'riplet_db/triplet_train/'

    # gen_triplet_data(triplet_db_path, triplet_data_save)
