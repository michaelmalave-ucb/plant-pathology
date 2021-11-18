import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop


img_height = 180
img_width = 180
CHANNELS = 3  # Keep RGB color channels to match the input format of the model
BATCH_SIZE = 34  # Big enough to measure an F1-score, a multiple of 18632
AUTOTUNE = tf.data.experimental.AUTOTUNE  # Adapt preprocessing and prefetching dynamically to reduce GPU and CPU idle time
cat_sample_size = 400
epochs = 20
pd.set_option('display.max_columns', None)

data_dir = "./data/train_images/"
labels_dir = "./data/train.csv"
categories = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust', 'scab']
num_classes = len(categories)

mlb = MultiLabelBinarizer(sparse_output=True)

image_data = pd.read_csv(labels_dir)
# shuffle data to start
image_data = image_data.sample(frac=1).reset_index(drop=True)
image_data['labels'] = image_data['labels'].apply(lambda x: str(x).split(" "))
image_data = image_data.join(
    pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(image_data.pop('labels')),
        index=image_data.index,
        columns=mlb.classes_))

sample_images = pd.DataFrame()
for c in categories:
    df_cat = image_data[image_data[c] == 1]
    df_images = df_cat[:cat_sample_size]
    sample_images = sample_images.append(df_images)

# overwrite image_data as the sampled data
print("number of images sampled:")
image_data = sample_images.drop_duplicates(subset='image', keep="last")
print(image_data.shape)

print("viewing data with one hot encoded labels")
print(image_data.head())


def get_image(image_file):
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_file, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [img_height, img_width])
    return image_resized


# preprocessing based on https://medium.com/deep-learning-with-keras/how-to-solve-multi-label-classification-problems-in-deep-learning-with-tensorflow-keras-7fb933243595
# https://www.tensorflow.org/tutorials/load_data/images
def get_label(filename):
    label = image_data[image_data['image'] == filename][categories].to_numpy().squeeze()
    label = tf.convert_to_tensor(label)
    return label


def process_path(file_path):
    label = get_label(file_path)
    image_path = data_dir + file_path
    img = tf.io.read_file(image_path)
    img = get_image(img)
    return img, label


def covert_onehot_string_labels(label_string, label_onehot):
    labels = []
    for i, label in enumerate(label_string):
        if label_onehot[i]:
            labels.append(label)
    if len(labels) == 0:
        labels.append("NONE")
    return labels


def show_samples(dataset):
    fig = plt.figure(figsize=(16, 16))
    columns = 3
    rows = 3
    print(columns * rows, "samples from the dataset")
    i = 1
    for a, b in dataset.take(columns * rows).cache():
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(a))
        plt.title("image shape:" + str(a.shape) + " (" + str(b.numpy()) + ") " +
                  str(covert_onehot_string_labels(categories, b.numpy())))
        i = i + 1
    plt.show()


def main():
    list_ds = tf.convert_to_tensor(image_data['image'])
    image_count = list_ds.shape[0]
    print("Dataset size: {}".format(list_ds.shape))
    numeric_dataset = tf.data.Dataset.from_tensor_slices(list_ds)
    # shuffle the data
    numeric_dataset = numeric_dataset.shuffle(image_count, reshuffle_each_iteration=False)

    train_ratio = 0.80
    ds_train = numeric_dataset.take(round(image_count * train_ratio))
    ds_test = numeric_dataset.skip(round(image_count * train_ratio))

    ds_train = ds_train.map(lambda x: tf.py_function(func=process_path,
                                                     inp=[x], Tout=(tf.float32, tf.int64)),
                            num_parallel_calls=tf.data.AUTOTUNE,
                            deterministic=False)
    ds_test = ds_test.map(lambda x: tf.py_function(func=process_path,
                                                   inp=[x], Tout=(tf.float32, tf.int64)),
                          num_parallel_calls=tf.data.AUTOTUNE,
                          deterministic=False)
    # show_samples(ds_test)
    ds_train_batched = ds_train.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)
    ds_test_batched = ds_test.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)

    print("Number of batches in train: ", ds_train_batched.cardinality().numpy())
    print("Number of batches in test: ", ds_test_batched.cardinality().numpy())

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # including transfer learning
    # https://www.tensorflow.org/guide/keras/transfer_learning
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(img_height, img_width, 3),  # VGG16 expects min 32 x 32
        include_top=False)  # Do not include the ImageNet classifier at the top.
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    scale_layer = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    activation = tf.keras.activations.sigmoid  # None  # tf.keras.activations.sigmoid or softmax

    outputs = tf.keras.layers.Dense(num_classes,
                                    kernel_initializer=initializer,
                                    activation=activation)(x)
    model = tf.keras.Model(inputs, outputs)

    """# create model with dropout
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation=activation),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation=activation),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation=activation),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation=activation),
        layers.Dense(num_classes)
    ])"""

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # default from_logits=False
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    history = model.fit(ds_train_batched, validation_data=ds_test_batched, epochs=epochs)

    ds = ds_test_batched
    print("Test Accuracy: ", model.evaluate(ds)[1])

    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    main()
