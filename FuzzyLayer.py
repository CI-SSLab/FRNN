#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Layer, InputSpec
import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform


class FuzzyLayer(Layer):
    
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.input_spec = InputSpec(min_ndim=2)
        super(FuzzyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.output_dim),
                                      initializer=RandomUniform(minval=0, maxval=1), trainable=True)
        super(FuzzyLayer, self).build(input_shape)

    def call(self, x, *args):
        s = []
        for i in range(x.shape[-1]):
            dd = []
            for j in range(self.output_dim):
                dij = tf.maximum(x[:, :, i], self.kernel[i, j])
                dd.append(tf.expand_dims(dij, axis=-1))
            s.append(tf.concat(dd, axis=-1))

        s = tf.concat(s, axis=-1)
        return s
