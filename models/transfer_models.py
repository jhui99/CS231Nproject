import tensorflow as tf
import os
import csv
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
    prediction_layer = tf.keras.layers.Dense(num_labels,activation='softmax')

    model = tf.keras.Sequential([
        VGG16_MODEL,
        global_average_layer,
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


def load_image(x, batch_folder):
    path = '../data/' + batch_folder + '/StreetViewImages/' + x + '.jpg'
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1 # convert to shape for VGG
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
 
    return image

def get_train_val_test(batch_folder, split=(70, 20, 10)):
    data_path = '../data/'
    batch_path = data_path + batch_folder + '/'2
    with open(data_path + 'labels.csv', 'r') as f:
        labels = list(csv.reader(f))
    X = []
    y = []
    for row in labels:
        X.append(row[0])
        y.append(int(row[1]))
    y = map_to_buckets(y)
    X = np.array(X)
    y = np.array(y)
    X_train, y_train, X_test_val, y_test_val = train_test_split(X, y, test_size=0.3)
    X_val, y_val, X_test, y_test = train_test_split(X_test_val, y_test_val, test_size=0.67) # 0.67 * 0.3 = 0.2

    def generator(X_, y_, batch_size=100):
        N = len(y_)
        i = 0
        ind = range(N) #indices
        while True:
            batch_X = []
            batch_y = []
            for b in range(batch_size):
                if i == N:
                    np.random.shuffle(ind)
                    i = 0
                i += 1
                curr_index = ind[i]
                x = X_[curr_index]
                x_image = load_image(x, batch_folder)
                batch_X.append(x_image)
                batch_y.append(y_[curr_index])
            
            batch_X = np.array(batch_X)
            batch_y = np.array(batch_y)
            yield (batch_X, batch_y)

    train_gen = generator(X_train, y_train)
    val_gen = generator(X_val, y_val)
    test_gen = generator(X_test, y_test)
    return train_gen, val_gen, test_gen

def train_and_eval(model, batch_folder, epochs, steps_per_epoch, validation_steps):
    history = model.fit(train_ds,  
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=5,
                    validation_data=validation_ds)
 
    loss0,accuracy0 = model.evaluate(validation_ds, steps = validation_steps)
    
    print("loss: {:.2f}".format(loss0))
    print("accuracy: {:.2f}".format(accuracy0))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def main():
    IMG_SIZE = (224, 224, 3)
    labels = ['low income', 'medium income', 'high income']
    model = get_vgg_transfer_model(IMG_SIZE, len(labels))
    train_and_eval(model=model, batch_folder='AZ_0.01stride', epochs=5, steps_per_epoch=2, validation_steps=2)

if __name__ == '__main__':
    main()


                

                