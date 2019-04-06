# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Ling Bao

@Function: main of cbir system
@date: 23/2/2019
"""

from common.query_faiss import FaissQuery


if __name__ == '__main__':
    """ main of cbir system """
    query_hdf5 = 'support/query.hdf5'
    h5_model_path = 'support/vgg16.h5'
    img_file = 'support/test.jpg'

    faiss_query = FaissQuery(query_hdf5, h5_model_path, k=1)
    distance, img_name = faiss_query.query(img_file)
    # distances, img_names = faiss_query.query_test(img_file)

    print('Distance = %.5f, Image path = %s' % (distance, img_name))
