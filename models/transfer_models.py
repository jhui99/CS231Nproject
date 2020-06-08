import tensorflow as tf
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing import image    

from tensorflow.keras.optimizers import (
    Adam
)

from vis.visualization import visualize_saliency
from vis.utils import utils

import seaborn as sn
import pandas as pd


from PIL import Image


#  THINGS TO TRY
# Conv-MaxPool-Dropout

def get_vgg_transfer_model(img_shape, num_labels):
    VGG16_MODEL=tf.keras.applications.VGG16(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')
    VGG16_MODEL.trainable=False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dense = tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    dense2 = tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    prediction_layer = tf.keras.layers.Dense(num_labels,activation='softmax', name='visualized_layer')

    model = tf.keras.Sequential([
        VGG16_MODEL,
        global_average_layer,
        dense,
        dense2,
        prediction_layer
    ])

    model.compile(optimizer=Adam(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

    return model

def get_vgg_transfer_model_reg(img_shape, num_labels):
    VGG16_MODEL=tf.keras.applications.VGG16(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')
    VGG16_MODEL.trainable=True

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dense = tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))
    dense2 = tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))
    prediction_layer = tf.keras.layers.Dense(num_labels,activation='softmax')

    model = tf.keras.Sequential([
        VGG16_MODEL,
        global_average_layer,
        dense,
        dense2,
        prediction_layer
    ])

    model.compile(optimizer=Adam(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

    return model

def map_to_buckets(y, loc):
    if loc =='AZ':
        b1, b2 = 60000, 100000
    elif loc == 'GA':
        b1, b2 = 26604, 75375
    elif loc == 'RI':
        b1, b2 = 50117, 65129
    elif loc == 'DC':
        b1, b2 = 38861, 93523
    elif loc == 'AA':
        b1, b2 = 63983, 79446
    elif loc == 'ZZ':
        b1, b2 = 75000, 150000
    def to_bucket(income):
        if income < b1:
            return 0
        elif income < b2:
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


def get_train_val_test(batch_folder, forced_loc, img_size, split=(70, 20, 10)):
    data_path = '../data/'
    if forced_loc != None:
        loc = forced_loc
    if 'AZ' in batch_folder:
        loc = 'AZ'
    elif 'GA' in batch_folder:
        loc = 'GA'
    elif 'RI' in batch_folder:
        loc = 'RI'
    elif 'DC' in batch_folder:
        loc = 'DC'
    elif 'AA' in batch_folder:
        loc = 'AA'
    batch_path = data_path + batch_folder + '/'
    with open(batch_path + 'labels.csv', 'r') as f:
        labels = list(csv.reader(f))
    X = []
    y = []
    for row in labels:
        X.append(row[0])
        y.append(float(row[1]))
    y = map_to_buckets(y, loc)
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
    return train_gen, val_gen, test_gen


def train_and_eval(model, img_size, batch_folder, epochs, steps_per_epoch, validation_steps, forced_loc=None, base_name=''):
    train_gen, val_gen, test_gen = get_train_val_test(batch_folder, forced_loc=forced_loc, img_size=img_size)
    history = model.fit(train_gen(),  
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=5,
                    validation_data=val_gen())  
 
    loss0,accuracy0 = model.evaluate(val_gen(), steps = validation_steps)

    name = base_name + batch_folder + '_' + str(epochs) + '_epochs'

    model.save_weights('weights/' + name + '_weights')
    
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


    # val_y = next(val_gen())
    # print(val_y)
    # y_pred = model.predict(val_y, steps=validation_steps)
    # y_pred = np.argmax(y_pred, axis=1)
    # print('pred shape', y_pred.shape)
    # print('Confusion Matrix')
    # print(confusion_matrix(val_gen().classes, y_pred))
    # print('Classification Report')
    # target_names = ['Low Income', 'Medium Income', 'High Income']
    # print(classification_report(val_gen().classes, y_pred, target_names=target_names))


def conf(model, x, y):
    pred = model.predict(x)
    pred = np.argmax(pred, axis=1)
    return tf.math.confusion_matrix(y, pred)



def confusion(img_size, batch_folder, epochs=100, forced_loc=None, base_name=''):
    print('start plot')
    name = base_name + batch_folder + '_' + str(epochs) + '_epochs'
    model = get_vgg_transfer_model((224, 224, 3), 3)
    model.load_weights('weights/' + name + '_weights')
    train_gen, val_gen, test_gen = get_train_val_test(batch_folder, forced_loc=forced_loc, img_size=img_size)

    DATA_SET_SIZES = {'GA_3_sat' : 6132, 'DC_1_sat' : 4252, 'RI_1_sat' : 3213}

    sz = DATA_SET_SIZES[batch_folder]
    num_batches = int((sz * 0.2) // 100)
    
    x, y = next(val_gen())
    conf_mat = conf(model, x, y)
    for i in range(num_batches):
        x, y = next(val_gen())
        conf_mat += conf(model, x, y)
    
    ind = ['Low', 'Medium', 'High']
    # remove diagonals since they are the # of correct assignments
    # conf_mat = tf.linalg.set_diag(conf_mat, [0, 0, 0])
    
    print('raw conf mat')
    print(conf_mat)
    conf_mat = conf_mat.numpy().astype('float32')
    row_sums = np.sum(conf_mat, axis=1)
    for i in range(3):
        conf_mat[i, :] = conf_mat[i, :] / row_sums[i]
    pd_conf = pd.DataFrame(conf_mat, index=ind, columns=ind)

    plt.clf()
    ax = plt.axes()
    sn.heatmap(pd_conf, annot=True)
    ax.set_title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    print('finished plot')
    print(conf_mat)
    print('row sums')
    print(np.sum(conf_mat, axis=1))
    plt.savefig('conf_mats/' + name + '_conf')
    plt.clf()

def saliency(img_size, batch_folder, epochs=50, forced_loc=None, base_name=''):
    name = base_name + batch_folder + '_' + str(epochs) + '_epochs'
    model = get_vgg_transfer_model((224, 224, 3), 3)
    model.load_weights('weights/' + name + '_weights')
    vis_path = 'vis/' + name
    if not os.path.isdir(vis_path):
        os.makedirs(vis_path)
    train_gen, val_gen, test_gen = get_train_val_test(batch_folder, forced_loc=forced_loc, img_size=img_size)

    # Find the index of the to be visualized layer above
    layer_index = utils.find_layer_idx(model, 'visualized_layer')

    # Swap softmax with linear
    model.layers[-1].activation = tf.keras.activations.linear

    labels = ['Low Income', 'Medium Income', 'High Income']
    # indices = [3000597, 3000526, 3001616, 3000395, 3003951, 3005799, 3005105, 3004200, 3005963]
    # input_classes = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    x, y = next(train_gen())
    pred = model.predict(x)
    pred = np.argmax(pred, axis=1)
    # Visualize
    for i in range(x.shape[0]):
        if i % 5 != 0:
            continue
        # Get input
        input_image = x[i]
        input_class = y[i]
        pred_class = pred[i]

        # Matplotlib preparations
        fig, axes = plt.subplots(1, 2)
        # Generate visualization
        visualization = visualize_saliency(model, layer_index, filter_indices=input_class, seed_input=input_image)

        axes[0].imshow(input_image[..., 0]) 
        axes[0].set_title('Original image')
        axes[1].imshow(visualization)
        axes[1].set_title('Saliency map')
        fig.suptitle(f'Correct class = {labels[input_class]}, Predicted = {labels[pred_class]}')
        plt.savefig(vis_path + '/' + str(i) + '.png')
        plt.clf()

def test_result(img_size, batch_folder, epochs=100, forced_loc=None, base_name='2d_60k_FINAL_dropout'):
    name = base_name + batch_folder + '_' + str(epochs) + '_epochs'
    model = get_vgg_transfer_model((224, 224, 3), 3)
    model.load_weights('weights/' + name + '_weights')

    train_gen, val_gen, test_gen = get_train_val_test(batch_folder, forced_loc=forced_loc, img_size=img_size)

    DATA_SET_SIZES = {'GA_3_sat' : 6132, 'DC_1_sat' : 4252, 'RI_1_sat' : 3213}

    sz = DATA_SET_SIZES[batch_folder]
    num_batches = int((sz * 0.1) // 100)
    corrects = 0
    seen = 0
    for i in range(num_batches + 1):
        x, y = next(val_gen())
        pred = model.predict(x)
        pred = np.argmax(pred, axis=1)
        corrects += np.sum(y == pred)
        seen += len(y)
    acc = corrects / seen
    print('test acc of {}: {}'.format(name, acc))


def main():
    img_size = 224
    batch_folders = ['GA_3_sat', 'DC_1_sat', 'RI_1_sat']
    labels = ['low income', 'medium income', 'high income']
    # for bf in batch_folders:
    #     model = get_vgg_transfer_model((224, 224, 3), len(labels))
    #     train_and_eval(model=model, img_size=img_size, batch_folder=bf, epochs=100, steps_per_epoch=16, validation_steps=2, forced_loc='AZ', base_name='2d_60k_FINAL_dropout')
    # for bf in batch_folders[:1]:
    #     model = get_vgg_transfer_model((224, 224, 3), len(labels))
    #     train_and_eval(model=model, img_size=img_size, batch_folder=bf, epochs=100, steps_per_epoch=16, validation_steps=2, forced_loc='ZZ', base_name='2d_75k_FINAL')
    # model = get_vgg_transfer_model((224, 224, 3), len(labels))
    # train_and_eval(model=model, img_size=img_size, batch_folder='GA_3_sat', epochs=50, steps_per_epoch=8, validation_steps=2, base_name='')
    # confusion_test(img_size, 'GA_3_sat')
    # saliency_test(img_size, 'GA_3_sat')
    # for bf in batch_folders:
        # confusion(img_size=img_size, batch_folder=bf, epochs=100, base_name='2d_60k_FINAL_dropout')
        # saliency(img_size, bf, epochs=100, forced_loc='AZ', base_name='2d_60k_FINAL_dropout')
        # test_result(img_size, bf, epochs=100, forced_loc='AZ', base_name='2d_60k_FINAL_dropout')

    # for bf in batch_folders:
    #     confusion(img_size=img_size, batch_folder=bf, epochs=100, base_name='2d_60k_FINAL_dropout')
    #     saliency(img_size, bf, epochs=100, forced_loc='AZ', base_name='2d_60k_FINAL_dropout')
        # test_result(img_size, bf, epochs=100, base_name='2d_FINAL_dropout')

    # for bf in batch_folders:
    #     confusion(img_size=img_size, batch_folder=bf, epochs=100, base_name='2d_FINAL_dropout')
    #     saliency(img_size, bf, epochs=100, base_name='2d_FINAL_dropout')
        # test_result(img_size, bf, epochs=100, base_name='2d_FINAL_dropout')

    
    # for bf in batch_folders:
    #     confusion(img_size=img_size, batch_folder=bf, epochs=100, base_name='2d_FINAL')
        # saliency(img_size, bf, epochs=100, base_name='2d_FINAL')
        # test_result(img_size, bf, epochs=100, base_name='2d_FINAL')

    for bf in batch_folders:
        # confusion(img_size=img_size, batch_folder=bf, epochs=100, base_name='2d_75k_FINAL')
        # saliency(img_size, bf, epochs=100, forced_loc='ZZ', base_name='2d_75k_FINAL')
        test_result(img_size, bf, epochs=100, forced_loc='ZZ', base_name='2d_75k_FINAL')




if __name__ == '__main__':
    main()


                

                