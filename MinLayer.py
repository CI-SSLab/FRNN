#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Layer, Minimum
import tensorflow as tf


class MinLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.min_layer = Minimum()
        super(MinLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MinLayer, self).build(input_shape)
        
    def call(self, inputs, *args):
        if type(inputs) != list:
            inputs = tf.unstack(inputs, axis=1)
        out = self.min_layer(inputs)
        return out
