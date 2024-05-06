import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.applications import MobileNetV2
from keras.optimizers import Adam

def build_model(input_shape, output_labels):
    # Load the pre-trained MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    # base arquitecture for model
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(output_labels, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model
