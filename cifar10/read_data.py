#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
created by Halo 2018/9/14 17:50
"""
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data = unpickle('./cifar-10-python/batches.meta')
print(data.keys())
