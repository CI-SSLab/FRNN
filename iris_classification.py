#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate
import numpy as np

from sklearn.preprocessing import StandardScaler

import FuzzyLayer
import MinLayer
import defuzzyfication
import GM
from sklearn import datasets
from sklearn.model_selection import train_test_split


def FRNN(seed=None):
    iris = datasets.load_iris()
    print("--- IRIS Loaded ---")

    x_data = iris['data']
    y_data = iris['target']

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled.astype(np.float32), y_data, test_size=0.1, random_state=seed)

    fuzzy_set_number = 5

    inp = Input(shape=(4,))

    h_out = GM.Gaussian1D(3)(inp)
    h1_out, h2_out, h3_out, h4_out = tf.unstack(h_out, axis=1)

    hb1 = FuzzyLayer.FuzzyLayer(fuzzy_set_number)
    p1 = hb1(tf.expand_dims(h1_out, 1))

    hb2 = FuzzyLayer.FuzzyLayer(fuzzy_set_number)
    p2 = hb2(tf.expand_dims(h2_out, 1))

    hb3 = FuzzyLayer.FuzzyLayer(fuzzy_set_number)
    p3 = hb3(tf.expand_dims(h3_out, 1))

    hb4 = FuzzyLayer.FuzzyLayer(fuzzy_set_number)
    p4 = hb4(tf.expand_dims(h4_out, 1))
    
    hb = concatenate([p1, p2, p3, p4], axis=1)

    hmin = MinLayer.MinLayer(1)(hb)
    
    defz = defuzzyfication.Defuzzy(3)
    out = defz(hmin)

    model = tf.keras.Model(inputs=[inp], outputs=[out])
    
    # loss categorical crossentropy
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=8, epochs=100)

    score = model.evaluate(x_test, y_test, batch_size=8)

    print("Loss: {} - Accuracy: {}".format(*score))
    
    
if __name__ == '__main__':
    FRNN()
