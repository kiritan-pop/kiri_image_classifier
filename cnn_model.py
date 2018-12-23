import os,glob,sys
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras import backend as tensorflow_backend
from tensorflow.keras.layers import BatchNormalization

def build_cnn_model(labels):
    # include_top=Falseによって、モデルから全結合層を削除
    xception_model = Xception(include_top=False, input_shape=(299, 299, 3), pooling='avg')
    layers = Dense(512)(xception_model.output)
    layers = Activation("relu")(layers)
    layers = GaussianNoise(0.3)(layers)
    layers = Dropout(0.3)(layers)
    layers = Dense(len(labels))(layers)
    layers = Activation("softmax")(layers)

    # 全結合層を削除したモデルと上で自前で構築した全結合層を結合
    model = Model(inputs=xception_model.inputs, outputs=layers)

    # レイヤーの重みを固定
    # print('model.layers:',len(xception_model.layers))
    # xception_model.trainable = False
    for layer in xception_model.layers[:-20]:
        layer.trainable = False
    return model
