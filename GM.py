#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:54:03 2020

@author: angelociaramella
"""

from tensorflow.keras.layers import Layer
import tensorflow as tf


class Gaussian1D(Layer):
    def __init__(self, truth_degrees, **kwargs):
        self.truth_degrees = truth_degrees
        super(Gaussian1D, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) == 1:
            shape_dim = input_shape[0][-1]
        else:
            shape_dim = input_shape[-1]
        self.c = self.add_weight(name='c', shape=(shape_dim, self.truth_degrees),
                                 initializer='uniform',
                                 trainable=True)
        self.a = self.add_weight(name='a', shape=(shape_dim, self.truth_degrees),
                                 initializer='ones',
                                 trainable=True)
        super(Gaussian1D, self).build(input_shape)

    def call(self, x, *args):
        aligned_x = tf.keras.backend.repeat_elements(tf.expand_dims(x, axis=-1), self.truth_degrees, -1)
        xc = tf.math.exp(-tf.square((aligned_x - self.c) / (2 * self.a)))
        return xc