# from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import Adam, Nadam
# from keras.callbacks import LambdaCallback,EarlyStopping,TensorBoard
# from keras.utils import multi_gpu_model, plot_model
# from keras import backend

import multiprocessing
import os,glob,sys,json
from cnn_model import build_cnn_model
import numpy as np
from PIL import Image
import argparse

import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--gpu", type=str, default='1')
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=96)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #パラメータ取得
    args = get_args()
    #GPU設定
    # config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False,
    #                                                   visible_device_list=args.gpu),
    #                         allow_soft_placement=True, 
    #                         log_device_placement=False
    #                         )
    # session = tf.Session(config=config)
    # tf.keras.backend.set_session(session)

    GPUs = len(args.gpu.split(','))
    start_idx = args.idx
    batch_size = args.batch_size
    model_path = args.model_path

    # 同時実行プロセス数
    process_count = multiprocessing.cpu_count()

    STANDARD_SIZE = (299, 299)
    # STANDARD_SIZE = (512, 512)
    epochs = 100000
    path_list = []
    image_list = []
    label_list = []
    img_dir = 'images/'
    # test_dir = 'test_images/'

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    # samplewise_center=True,
                    # samplewise_std_normalization=True,
                    # zca_whitening=True,   # unknown
                    rotation_range=360, # 90°まで回転
                    width_shift_range=0.4, # 水平方向にランダムでシフト
                    height_shift_range=0.4, # 垂直方向にランダムでシフト
                    #channel_shift_range=50.0, # 色調をランダム変更
                    shear_range=0.3, # 斜め方向(pi/8まで)に引っ張る
                    horizontal_flip=True, # 垂直方向にランダムで反転
                    vertical_flip=True, # 水平方向にランダムで反転
                    zoom_range=0.5,
                    validation_split=0.1,
                    fill_mode='reflect'
                    )

    # train_datagen.fit(xxxx)

    # 画像の拡張
    train_generator = train_datagen.flow_from_directory(
        img_dir,
        batch_size=batch_size,
        # save_to_dir="temp/",
        target_size=STANDARD_SIZE,
        subset="training")

    validation_generator = train_datagen.flow_from_directory(
        img_dir,
        batch_size=batch_size,
        target_size=STANDARD_SIZE,
        subset="validation")

    print(train_generator.class_indices)
    with open('.cnn_labels','w') as fw:
        json.dump(train_generator.class_indices,fw,indent=4)

    # モデルを読み込む
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        model = build_cnn_model(train_generator.class_indices)

    def on_epoch_end(epoch, logs):
        model.save(model_path)
        model.save( str(epoch) + model_path)

    print_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
    ES = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
    # TB = TensorBoard(histogram_freq=1)
    # TB = TensorBoard(write_grads=True,write_images=3, histogram_freq=1)

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png')
    m = model
    m.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])

    m.fit_generator(
            train_generator,
            callbacks=[print_callback,ES],
            steps_per_epoch=500,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=10,
            initial_epoch=start_idx,
            max_queue_size=process_count,
            workers=6,
            use_multiprocessing=False)
