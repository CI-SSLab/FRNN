from tensorflow.keras.layers import Layer, Flatten, Dense
import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform


class Defuzzy(Layer):
    def __init__(self, output_dim, normalization=True, use_bias=True, **kwargs):
        self.output_dim = output_dim
        self.normalization = normalization
        self.function = Dense(self.output_dim, use_bias=use_bias,
                              kernel_initializer=RandomUniform(minval=0, maxval=1))
              
        super(Defuzzy, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(Defuzzy, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, *args):
        x = Flatten()(inputs)
        logits = self.function(x)
        return logits if not self.normalization else \
            logits / tf.reduce_sum(x, axis=-1, keepdims=True)
