import os,glob,sys
# import keras
import tensorflow.keras as keras


def alt_build_cnn_model(labels):
    d_out = 0.0
    input_image = keras.layers.Input(shape=(299, 299, 3))

    model = keras.layers.GaussianNoise(0.01)(input_image)
    model = keras.layers.Conv2D(filters=64,  kernel_size=3, strides=1, padding='same')(model)
    model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    GAP0  = keras.layers.GlobalAveragePooling2D()(model)
 
    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=96,  kernel_size=4, strides=2, padding='same')(model) # >64
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    # GAP11  = keras.layers.GlobalAveragePooling2D()(model)
    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=96,  kernel_size=3, strides=1, padding='same')(model)
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    GAP12  = keras.layers.GlobalAveragePooling2D()(model)
 
    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=128,  kernel_size=4, strides=2, padding='same')(model) # >32
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    # GAP21  = keras.layers.GlobalAveragePooling2D()(model)
    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=128,  kernel_size=3, strides=1, padding='same')(model)
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    GAP22  = keras.layers.GlobalAveragePooling2D()(model)
 
    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=192,  kernel_size=4, strides=2, padding='same')(model) # >32
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    # GAP31  = keras.layers.GlobalAveragePooling2D()(model)
    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=192,  kernel_size=3, strides=1, padding='same')(model)
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    GAP32 = keras.layers.GlobalAveragePooling2D()(model)
 
    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=256,  kernel_size=4, strides=2, padding='same')(model) # >32
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    # GAP41  = keras.layers.GlobalAveragePooling2D()(model)
    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=256,  kernel_size=3, strides=1, padding='same')(model)
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    GAP42 = keras.layers.GlobalAveragePooling2D()(model)

    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=384,  kernel_size=4, strides=2, padding='same')(model) # >32
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    # GAP51  = keras.layers.GlobalAveragePooling2D()(model)
    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=384,  kernel_size=3, strides=1, padding='same')(model)
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    GAP52 = keras.layers.GlobalAveragePooling2D()(model)

    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=512,  kernel_size=4, strides=2, padding='same')(model) # >32
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    # GAP61  = keras.layers.GlobalAveragePooling2D()(model)
    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Conv2D(filters=512,  kernel_size=3, strides=1, padding='same')(model)
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.LeakyReLU()(model)
    GAP62 = keras.layers.GlobalAveragePooling2D()(model)

    model = keras.layers.Concatenate()([GAP0, GAP12, GAP22, GAP32, GAP42, GAP52, GAP62])
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    # model = keras.layers.Dense(512)(model)
    # model = keras.layers.BatchNormalization(momentum=0.8)(model)
    model = keras.layers.Dropout(d_out)(model)
    model = keras.layers.Dense(len(labels))(model)
    model = keras.layers.Softmax()(model)
    return keras.models.Model(inputs=input_image, outputs=model)

def build_cnn_model(labels):
    # include_top=Falseによって、モデルから全結合層を削除
    base_model = keras.applications.InceptionV3(include_top=False, input_shape=(299, 299, 3), pooling='avg')
    pred = keras.layers.Dense(len(labels), activation="softmax")(base_model.output)

    # 全結合層を削除したモデルと上で自前で構築した全結合層を結合
    model = keras.models.Model(inputs=base_model.inputs, outputs=pred)

    # レイヤーの重みを固定
    for layer in base_model.layers[:50]:
        layer.trainable = False
    return model
