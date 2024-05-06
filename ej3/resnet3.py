 
import tensorflow as tf
from keras.layers import Input, RandomFlip, RandomContrast, BatchNormalization, RandomTranslation, Flatten, Dropout
from keras.layers import Add,Dense, Conv2D, Activation, MaxPooling2D, GlobalMaxPooling2D, Rescaling

def residual_block(x, number_of_filters, kernel, match_filter_size=False):
    """
    Residual block with
    """
    x_skip = x


    x = Conv2D(number_of_filters, kernel_size=kernel, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(number_of_filters, kernel_size=kernel, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    
    x_skip = Conv2D(number_of_filters, kernel_size=kernel)(x_skip)
    x_skip = BatchNormalization()(x)
    # Add the skip connection to the regular mapping
    print(x.shape, x_skip.shape)
    x = Add()([x, x_skip])
    x = MaxPooling2D((2, 2))(x)
    # Nonlinearly activate the result
    x = Activation("relu")(x)

    # Return the result
    return x


def build_model(input_shape, output_labels, filters_and_kernels=[], match_filter_size=False):

    filters_and_kernels = filters_and_kernels or [
        (64,  (3,3)), 
        (128, (3,3)),
        (256, (3,3)), 
        (512, (3,3))
    ]

    i = Input(input_shape, dtype=tf.float32)

    x = Rescaling(1./255)(i)
    x = RandomFlip("horizontal")(x)
    x = RandomFlip("vertical")(x)
    x = RandomTranslation(0.1, 0.1, fill_mode="reflect")(x)
    x = RandomContrast(0.2)(x)

    for number_of_filters, kernel in filters_and_kernels:
        x = residual_block(x, number_of_filters, kernel, match_filter_size=match_filter_size)

    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(output_labels)(x)
    x = Activation("softmax")(x)
    return tf.keras.Model(inputs=[i], outputs=[x])