import os,glob,sys
# from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras import backend as tensorflow_backend
from tensorflow.keras.layers import BatchNormalization

def build_cnn_model(labels):
    # include_top=Falseによって、モデルから全結合層を削除
    # base_model = Xception(include_top=False, input_shape=(299, 299, 3), pooling='avg')
    input_img = Input(shape=(299,299,3))
    noisy_img = GaussianNoise(0.05)(input_img)
    base_model = InceptionV3(include_top=False, pooling='avg', input_tensor=noisy_img)
    pred = Dropout(0.3)(base_model.output)
    pred = Dense(len(labels), activation="softmax")(pred)

    # 全結合層を削除したモデルと上で自前で構築した全結合層を結合
    model = Model(inputs=input_img, outputs=pred)

    # レイヤーの重みを固定
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    return model
