""" Create Inception model architecture."""
import tensorflow as tf

import modules

def build_model():
    """ Build and return Tensorflow model with Inception architecure."""
    # Input layer.
    input_layer = tf.keras.layers.Input(shape=(299,299,3), name='input_layer')

    # Stem module.
    stem = modules.InceptionStemBlock(name='stem_block')(input_layer)

    # Inception modules.
    # Four inception blocks and reduction.
    inception_1a = modules.InceptionBlockA(name='inception_blocka1')(stem)
    inception_1b = modules.InceptionBlockA(name='inception_blocka2')(inception_1a)
    inception_1c = modules.InceptionBlockA(name='inception_blocka3')(inception_1b)
    inception_1d = modules.InceptionBlockA(name='inception_blocka4')(inception_1c)
    inception_1_reduced = modules.InceptionReductionA(name='blocka_reduce')(inception_1d)

    # Seven inception blocks and reduction.
    inception_2a = modules.InceptionBlockB(name='inception_blockb1')(inception_1_reduced)
    inception_2b = modules.InceptionBlockB(name='inception_blockb2')(inception_2a)
    inception_2c = modules.InceptionBlockB(name='inception_blockb3')(inception_2b)
    inception_2d = modules.InceptionBlockB(name='inception_blockb4')(inception_2c)
    inception_2e = modules.InceptionBlockB(name='inception_blockb5')(inception_2d)
    inception_2f = modules.InceptionBlockB(name='inception_blockb6')(inception_2e)
    inception_2g = modules.InceptionBlockB(name='inception_blockb7')(inception_2f)
    inception_2_reduced = modules.InceptionReductionB(name='inception_blockb_reduce')(inception_2g)

    # Three inception blocks.
    inception_3a = modules.InceptionBlockC(name='inception_blockc1')(inception_2_reduced)
    inception_3b = modules.InceptionBlockC(name='inception_blockc2')(inception_3a)
    inception_3c = modules.InceptionBlockC(name='inception_blockc3')(inception_3b)

    # Average pooling and dropout layers.
    averagepool_layer = tf.keras.layers.AveragePooling2D(
        pool_size=(8,8),
        strides=(1,1),
        name='inception_averagepool'
    )(inception_3c)
    dropout_layer = tf.keras.layers.Dropout(rate=0.2, name='dropout_layer')(averagepool_layer)
    flatten_layer = tf.keras.layers.Flatten()(dropout_layer)

    # Output.
    output_layer = tf.keras.layers.Dense(
        units=1000,
        activation='softmax',
        name='output_layer'
    )(flatten_layer)

    inception_model = tf.keras.Model(inputs=input_layer, outputs=[output_layer])

    return inception_model
