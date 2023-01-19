""" Module blocks needed for model construction."""
import tensorflow as tf

'Basic convolutional block.'
class ConvolutionalBlock(tf.keras.Model):
    """ 2D convolutional block with batch normalization and relu activaiton."""
    def __init__(self, **kwargs):
        """ Initialize layers."""
        super(ConvolutionalBlock, self).__init__(name='')
        self.conv_layer = tf.keras.layers.Conv2D(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_conv = self.conv_layer(input_tensor)
        x_normalized = self.batch_norm(x_conv)
        x_output = self.activation(x_normalized)

        return x_output

'Inception blocks.'
class InceptionBlockA(tf.keras.Model):
    """ Traditional inception block."""
    def __init__(self, name=''):
        """ Initialize layers."""
        super(InceptionBlockA, self).__init__(name=name)
        # Branch with two layers.
        self.branch0_0 = ConvolutionalBlock(
            filters=64,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )
        self.branch0_1 = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same'
        )

        # Branch with three layers.
        self.branch1_0 = ConvolutionalBlock(
            filters=64,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )
        self.branch1_1 = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same'
        )
        self.branch1_2 = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same'
        )
        # Branch with two layers.
        self.branch2_0 = tf.keras.layers.AveragePooling2D(
            pool_size=(3,3),
            strides=(1,1),
            padding='same'
        )
        self.branch2_1 = ConvolutionalBlock(
            filters=96,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )
        # Single 1x1 convolutional layer.
        self.branch3 = ConvolutionalBlock(
            filters=96,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )

        self.output_layer = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_0 = self.branch0_0(input_tensor)
        x_0 = self.branch0_1(x_0)

        x_1 = self.branch1_0(input_tensor)
        x_1 = self.branch1_1(x_1)
        x_1 = self.branch1_2(x_1)

        x_2 = self.branch2_0(input_tensor)
        x_2 = self.branch2_1(x_2)

        x_3 = self.branch3(input_tensor)

        x_output = self.output_layer([ x_0, x_1, x_2, x_3 ])

        return x_output

class InceptionBlockB(tf.keras.Model):
    """ Computationally efficient factorization of nxn convolutional layer."""
    def __init__(self, name=''):
        """ Initialize layers."""
        super(InceptionBlockB, self).__init__(name=name)
        # Branch with five layers.
        self.branch0_0 = ConvolutionalBlock(
            filters=192,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )
        self.branch0_1 = ConvolutionalBlock(
            filters=192,
            kernel_size=(1,7),
            strides=(1,1),
            padding='same'
        )
        self.branch0_2 = ConvolutionalBlock(
            filters=224,
            kernel_size=(7,1),
            strides=(1,1),
            padding='same'
        )
        self.branch0_3 = ConvolutionalBlock(
            filters=224,
            kernel_size=(1,7),
            strides=(1,1),
            padding='same'
        )
        self.branch0_4 = ConvolutionalBlock(
            filters=256,
            kernel_size=(7,1),
            strides=(1,1),
            padding='same'
        )
        # Branch with three layers.
        self.branch1_0 = ConvolutionalBlock(
            filters=192,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )
        self.branch1_1 = ConvolutionalBlock(
            filters=224,
            kernel_size=(1,7),
            strides=(1,1),
            padding='same'
        )
        self.branch1_2 = ConvolutionalBlock(
            filters=256,
            kernel_size=(7,1),
            strides=(1,1),
            padding='same'
        )
        # Branch with two layers.
        self.branch2_0 = tf.keras.layers.AveragePooling2D(
            pool_size=(3,3),
            strides=(1,1),
            padding='same'
        )
        self.branch2_1 = ConvolutionalBlock(
            filters=128,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )
        # Branch with single 1x1 layer.
        self.branch3 = ConvolutionalBlock(
            filters=384,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )

        self.output_layer = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_0 = self.branch0_0(input_tensor)
        x_0 = self.branch0_1(x_0)
        x_0 = self.branch0_2(x_0)
        x_0 = self.branch0_3(x_0)
        x_0 = self.branch0_4(x_0)

        x_1 = self.branch1_0(input_tensor)
        x_1 = self.branch1_1(x_1)
        x_1 = self.branch1_2(x_1)

        x_2 = self.branch2_0(input_tensor)
        x_2 = self.branch2_1(x_2)

        x_3 = self.branch3(input_tensor)

        x_output = self.output_layer([x_0, x_1, x_2, x_3])

        return x_output

class InceptionBlockC(tf.keras.Model):
    """ Inception module with expanded filter bank."""
    def __init__(self, name=''):
        """ Initialize layers."""
        super(InceptionBlockC, self).__init__(name=name)
        # Branch with three stem layers and two separate output layers.
        self.branch0_0 = ConvolutionalBlock(
            filters=384,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )
        self.branch0_1 = ConvolutionalBlock(
            filters=448,
            kernel_size=(1,3),
            strides=(1,1),
            padding='same'
        )
        self.branch0_2 = ConvolutionalBlock(
            filters=512,
            kernel_size=(3,1),
            strides=(1,1),
            padding='same'
        )
        self.branch0a = ConvolutionalBlock(
            filters=256,
            kernel_size=(3,1),
            strides=(1,1),
            padding='same'
        )
        self.branch0b = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,3),
            strides=(1,1),
            padding='same'
        )
        # Branch with one stem layer and two output layers.
        self.branch1_0 = ConvolutionalBlock(
            filters=384,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )
        self.branch1a = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,3),
            strides=(1,1),
            padding='same'
        )
        self.branch1b = ConvolutionalBlock(
            filters=256,
            kernel_size=(3,1),
            strides=(1,1),
            padding='same'
        )
        # Branch with two layers.
        self.branch2_0 = tf.keras.layers.AveragePooling2D(
            pool_size=(3,3),
            strides=(1,1),
            padding='same'
        )
        self.branch2_1 = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )
        # Single 1x1 convolutional layer.
        self.branch3 = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )

        self.output_layer = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_branch0 = self.branch0_0(input_tensor)
        x_branch0 = self.branch0_1(x_branch0)
        x_branch0 = self.branch0_2(x_branch0)

        x_branch0a = self.branch0a(x_branch0)
        x_branch0b = self.branch0b(x_branch0)

        x_branch1 = self.branch1_0(input_tensor)

        x_branch1a = self.branch1a(x_branch1)
        x_branch1b = self.branch1b(x_branch1)

        x_branch2 = self.branch2_0(input_tensor)
        x_branch2 = self.branch2_1(x_branch2)

        x_branch3 = self.branch3(input_tensor)

        x_output = self.output_layer(
            [
                x_branch0a, x_branch0b,
                x_branch1a, x_branch1b,
                x_branch2,
                x_branch3
            ]
        )

        return x_output

'Dimension reduction blocks.'
class InceptionReductionA(tf.keras.Model):
    """ Dimension reduction block."""
    def __init__(self, name=''):
        """ Initialize layers."""
        super(InceptionReductionA, self).__init__(name=name)
        # Branch with three layers.
        self.branch0_0 = ConvolutionalBlock(
            filters=192,
            kernel_size=(1,1)
        )
        self.branch0_1 = ConvolutionalBlock(
            filters=224,
            kernel_size=(3,3),
            strides=(1,1)
        )
        self.branch0_2 = ConvolutionalBlock(
            filters=256,
            kernel_size=(3,3),
            strides=(2,2),
            padding='same'
        )
        # Branch with single 3x3 convolutional layer.
        self.branch1 = ConvolutionalBlock(
            filters=384,
            kernel_size=(3,3),
            strides=(2,2),
            padding='same'
        )
        # Branch with single maxpooling layer.
        self.branch2 = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='same'
        )

        self.output_layer = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_branch0 = self.branch0_0(input_tensor)
        x_branch0 = self.branch0_1(input_tensor)
        x_branch0 = self.branch0_2(input_tensor)

        x_branch1 = self.branch1(input_tensor)

        x_branch2 = self.branch2(input_tensor)

        x_output = self.output_layer([x_branch0, x_branch1, x_branch2])

        return x_output

class InceptionReductionB(tf.keras.Model):
    """ Dimension reducton block."""
    def __init__(self, name=''):
        """ Initialize layers."""
        super(InceptionReductionB, self).__init__(name=name)
        # Branch with four layers.
        self.branch0_0 = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )
        self.branch0_1 = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,7),
            strides=(1,1),
            padding='same'
        )
        self.branch0_2 = ConvolutionalBlock(
            filters=320,
            kernel_size=(7,1),
            strides=(1,1),
            padding='same'
        )
        self.branch0_3 = ConvolutionalBlock(
            filters=320,
            kernel_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
        # Branch with two layers.
        self.branch1_0 = ConvolutionalBlock(
            filters=192,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same'
        )
        self.branch1_1 = ConvolutionalBlock(
            filters=192,
            kernel_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
        # Branch with single maxpooling layer.
        self.branch2 = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )

        self.output_layer = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_branch0 = self.branch0_0(input_tensor)
        x_branch0 = self.branch0_1(x_branch0)
        x_branch0 = self.branch0_2(x_branch0)
        x_branch0 = self.branch0_3(x_branch0)

        x_branch1 = self.branch1_0(input_tensor)
        x_branch1 = self.branch1_1(x_branch1)

        x_branch2 = self.branch2(input_tensor)

        x_output = self.output_layer([ x_branch0, x_branch1, x_branch2 ])

        return x_output

'Stem block before inception layers.'
class InceptionStemBlock(tf.keras.Model):
    """ Initial layers before """
    def __init__(self, name=''):
        """ Initialize layers."""
        super(InceptionStemBlock, self).__init__(name=name)

        self.stem_conv1 = ConvolutionalBlock(
            input_shape=(299,299,3),
            filters=32,
            kernel_size=(3,3),
            strides=(2,2),
            padding='valid',
            name='stem_conv1'
        )
        self.stem_conv2 = ConvolutionalBlock(
            filters=32,
            kernel_size=(3,3),
            strides=(1,1),
            padding='valid',
            name='stem_conv2'
        )
        self.stem_conv3 = ConvolutionalBlock(
            filters=64,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            name='stem_conv3'
        )

        self.stem_branch1a = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid',
            name='stem_branch1a'
        )
        self.stem_branch1b = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3),
            strides=(2,2),
            padding='valid',
            name='stem_branch1b'
        )
        self.stem_concat1 = tf.keras.layers.Concatenate(axis=-1, name='stem_concat1')

        self.stem_branch2a_1 = ConvolutionalBlock(
            filters=64,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same',
            name='stem_branch2a_1'
        )
        self.stem_branch2a_2 = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3),
            strides=(1,1),
            padding='valid',
            name='stem_branch2a_2'
        )
        self.stem_branch2b_1 = ConvolutionalBlock(
            filters=64,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same',
            name='stem_branch2b_1'
        )
        self.stem_branch2b_2 = ConvolutionalBlock(
            filters=64,
            kernel_size=(7,1),
            strides=(1,1),
            padding='same',
            name='stem_branch2b_2'
        )
        self.stem_branch2b_3 = ConvolutionalBlock(
            filters=64,
            kernel_size=(1,7),
            strides=(1,1),
            padding='same',
            name='stem_branch2b_3'
        )
        self.stem_branch2b_4 = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3),
            padding='valid',
            strides=(1,1),
            name='stem_branch2b_4'
        )
        self.stem_concat2 = tf.keras.layers.Concatenate(axis=-1, name='stem_concat2')

        self.stem_branch3a = ConvolutionalBlock(
            filters=192,
            kernel_size=(3,3),
            padding='same',
            strides=(2,2),
            name='stem_branch3a'
        )
        self.stem_branch3b = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='same',
            name='stem_branch3b'
        )

        self.output_layer = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_conv1 = self.stem_conv1(input_tensor)
        x_conv2 = self.stem_conv2(x_conv1)
        x_conv3 = self.stem_conv3(x_conv2)

        x_branch1a = self.stem_branch1a(x_conv3)
        x_branch1b = self.stem_branch1b(x_conv3)

        x_branch1 = self.stem_concat1([x_branch1a, x_branch1b])

        x_branch2a = self.stem_branch2a_1(x_branch1)
        x_branch2a = self.stem_branch2a_2(x_branch2a)

        x_branch2b = self.stem_branch2b_1(x_branch1)
        x_branch2b = self.stem_branch2b_2(x_branch2b)
        x_branch2b = self.stem_branch2b_3(x_branch2b)
        x_branch2b = self.stem_branch2b_4(x_branch2b)

        x_branch2 = self.stem_concat2([x_branch2a, x_branch2b])

        x_branch3a = self.stem_branch3a(x_branch2)
        x_branch3b = self.stem_branch3b(x_branch2)

        x_output = self.output_layer([x_branch3a, x_branch3b])

        return x_output
