import tensorflow as tf
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing import image    

from tensorflow.keras.optimizers import (
    Adam
)

def get_vgg_transfer_model(img_shape, num_labels):
    VGG16_MODEL=tf.keras.applications.VGG16(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')
    VGG16_MODEL.trainable=False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dense = tf.keras.layers.Dense(150, activation='relu')
    prediction_layer = tf.keras.layers.Dense(num_labels,activation='softmax')
    conv1 = tf.keras.layers.Conv2D(20, (7,7), strides=(1, 1), input_shape=(224, 224, 3))
    conv2 = tf.keras.layers.Conv2D(20, (3,3), strides=(2, 2))
    maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    model = tf.keras.Sequential([
        # VGG16_MODEL,
        conv1,
        maxpool,
        conv2,
        maxpool,
        conv2,
        global_average_layer,
        dense,
        prediction_layer
    ])

    model.compile(optimizer=Adam(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

    return model

def map_to_buckets(y):
    def to_bucket(income):
        if income < 60000:
            return 0
        elif income < 100000:
            return 1
        return 2
    return list(map(to_bucket, y))


def load_image(x, img_size, batch_path):
    path = batch_path + 'Images/' + x + '.jpg'
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1 # convert to shape for VGG
    image = tf.image.resize(image, (img_size, img_size))
 
    return image

def duplicate_angles(X, y):
    y = y.repeat(6)
    X = X.repeat(6)
    new_X = []
    for i in range(len(X)):
        new_str = str(X[i]) + '-' + str(60 * (i % 6))
        new_X.append(new_str)
    new_X = np.array(new_X)
    return new_X, y


def get_train_val_test(batch_folder, img_size, split=(70, 20, 10)):
    data_path = '../data/'
    batch_path = data_path + batch_folder + '/'
    with open(batch_path + 'labels.csv', 'r') as f:
        labels = list(csv.reader(f))
    X = []
    y = []
    for row in labels:
        X.append(row[0])
        y.append(float(row[1]))
    y = map_to_buckets(y)
    X = np.array(X)
    y = np.array(y)
    X_train, X_test_val, y_train,  y_test_val = train_test_split(X, y, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.67) # 0.67 * 0.3 = 0.2
    if batch_folder[-3:] != 'sat':
        X_train, y_train = duplicate_angles(X_train, y_train)
        X_val, y_val = duplicate_angles(X_val, y_val)
        X_test, y_test = duplicate_angles(X_test, y_test)

    def generator(X_, y_, batch_size=100):
        def gen():
            N = len(y_)
            i = 0
            ind = np.arange(N) #indices
            while True:
                batch_X = []
                batch_y = []
                for b in range(batch_size):
                    if i == N:
                        np.random.shuffle(ind)
                        i = 0
                    curr_index = ind[i]
                    x = X_[curr_index]
                    x_image = load_image(x, img_size, batch_path)
                    batch_X.append(x_image)
                    batch_y.append(y_[curr_index])
                    i += 1
                batch_X = np.stack(batch_X)
                batch_y = np.array(batch_y)
                yield (batch_X, batch_y)
        return gen

    train_gen = generator(X_train, y_train)
    val_gen = generator(X_val, y_val)
    test_gen = generator(X_test, y_test)
    # X_train_ = []
    # for x in X_train:
    #     img = load_image(x, img_size, batch_path)
    #     X_train_.append(np.expand_dims(image, axis=0))
    #     if 
    # train = tf.data.Dataset.from_tensor_slices((X_train_, y_train))
    # train.batch(100)
    # train = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float32, tf.int16), output_shapes=(tf.TensorShape((100, 224, 224, 3)), (100,)))
    # val = tf.data.Dataset.from_generator(val_gen, output_types=(tf.float32, tf.int16), output_shapes=(tf.TensorShape((100, 224, 224, 3)), (100,)))
    # test = tf.data.Dataset.from_generator(test_gen, output_types=(tf.float32, tf.int16))
    return train_gen, val_gen, test_gen


def train_and_eval(model, img_size, batch_folder, epochs, steps_per_epoch, validation_steps, base_name=''):
    train_gen, val_gen, test = get_train_val_test(batch_folder, img_size)
    history = model.fit(train_gen(),  
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=5,
                    validation_data=val_gen())  
 
    loss0,accuracy0 = model.evaluate(val_gen(), steps = validation_steps)

    name = base_name + batch_folder + '_' + str(epochs) + '_epochs'
    
    print(name + "loss: {:.2f}".format(loss0))
    print(name + "accuracy: {:.2f}".format(accuracy0))
    
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accs/' + name + '_acc.png')

    plt.clf()
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('losses/' + name + '_loss.png')

def main():
    img_size = 224
    batch_folders = ['GA_3_sat', 'DC_1_sat', 'RI_1_sat']
    labels = ['low income', 'medium income', 'high income']
    for bf in batch_folders:
        model = get_vgg_transfer_model((224, 224, 3), len(labels))
        train_and_eval(model=model, img_size=img_size, batch_folder=bf, epochs=50, steps_per_epoch=2, validation_steps=2)

if __name__ == '__main__':
    main()


                

                