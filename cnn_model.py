import os,glob,sys
import tensorflow as tf

def build_cnn_model(labels):
    # include_top=Falseによって、モデルから全結合層を削除
    # input_image = keras.layers.Input(shape=(299, 299, 3))
    # input = keras.layers.GaussianNoise(0.2)(input_image)
    # base_model = keras.applications.InceptionV3(include_top=False, input_shape=(299, 299, 3), pooling='avg')
    # base_model = keras.applications.InceptionV3(include_top=False, input_tensor=input, pooling='avg')
    # base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet')
    base_model = tf.keras.applications.Xception(include_top=False, pooling='avg')
    x = base_model.output
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    pred = tf.keras.layers.Dense(len(labels), activation="softmax")(x)

    # 全結合層を削除したモデルと上で自前で構築した全結合層を結合
    model = tf.keras.models.Model(inputs=base_model.input, outputs=pred)

    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # レイヤーの重みを固定
    num=len(model.layers)//2
    for layer in model.layers[:num]:
        layer.trainable = False
    for layer in model.layers[num:]:
        layer.trainable = True
    return model
