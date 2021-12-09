import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MultiLabelBinarizer

# tensorflow config
tf.config.threading.set_inter_op_parallelism_threads(16)


img_height = 224  # input shape has to be (331, 331, 3) for NASNetLarge
img_width = 224

CHANNELS = 3  # Keep RGB color channels to match the input format of the model
BATCH_SIZE = 50  # Big enough to measure an F1-score, a multiple of 18632
AUTOTUNE = tf.data.experimental.AUTOTUNE  # Adapt preprocessing and prefetching dynamically to reduce GPU and CPU idle time
cat_sample_size = 400

epochs = 20

pd.set_option('display.max_columns', None)
base_learning_rate = 0.001

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
image_data = sample_images.drop_duplicates(subset='image', keep="last")


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

    ds_train_batched = ds_train.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    ds_test_batched = ds_test.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

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
            layers.RandomContrast(0.2),

        ]
    )

    # including transfer learning
    # https://www.tensorflow.org/guide/keras/transfer_learning

    base_model = tf.keras.applications.MobileNetV2(  # configuration might be updated
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(img_height, img_width, 3),  # VGG16 expects min 32 x 32
        include_top=False)  # Do not include the ImageNet classifier at the top.
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = tf.keras.layers.GaussianNoise(0.2)(x)
    scale_layer = tf.keras.layers.Rescaling(scale=1. / 127.5, offset=-1)
    x = scale_layer(x)
    x = base_model(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    initializer = tf.keras.initializers.HeUniform(seed=42)  # configuration might be updated

    activation = tf.keras.activations.softmax  # None  # tf.keras.activations.sigmoid or softmax # configuration might be updated

    outputs = tf.keras.layers.Dense(num_classes,
                                    kernel_initializer=initializer,
                                    activation=activation)(x)

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
        tf.keras.metrics.AUC(
            num_thresholds=200, curve='ROC',
            summation_method='interpolation', name='roc', dtype=None,
            thresholds=None, multi_label=True, num_labels=num_classes, label_weights=None,
            from_logits=True
        )
    ]

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # default from_logits=False
                  metrics=[METRICS])
    history = model.fit(ds_train_batched, validation_data=ds_test_batched, epochs=epochs)

    model.summary()

    ds = ds_test_batched
    epochs_range = range(epochs)
    # Store metrics
    test_acc = round(model.evaluate(ds)[1], 4)
    test_auc = round(model.evaluate(ds)[-1], 4)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print("Test Accuracy: ", test_acc)
    print("Test AUC Value: ", test_auc)

    true_pos = history.history['tp']
    false_pos = history.history['fp']
    true_neg = history.history['tn']
    false_neg = history.history['fn']
    auc = history.history['auc']
    prc = history.history['prc']
    roc = history.history['roc']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Get Model Statistics
    # calculations based off https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
    tp = np.array(history.history['tp'])
    fp = np.array(history.history['fp'])
    tn = np.array(history.history['tn'])
    fn = np.array(history.history['fn'])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    # plot accuracy, loss, and roc
    plt.figure(figsize=(16, 16))
    plt.subplot(3, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(3, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.subplot(3, 1, 3)
    plt.plot(epochs_range, auc, label='auc')
    plt.title('Area Under ROC (AUC) Over Epochs')
    plt.legend(loc='upper right')
    plt.show()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tp / (tn + fp)
    misclass = (fp + fn) / (tp + tn + fp + fn)
    F1 = 2 * [(precision * recall) / (precision + recall)][-1]

    print("accuracy: " + str(round(accuracy[-1], 4)))
    print("precision: " + str(round(precision[-1], 4)))
    print("recall: " + str(round(recall[-1], 4)))
    print("specificity: " + str(round(specificity[-1], 4)))
    print("misclass: " + str(round(misclass[-1], 4)))
    print("F1: " + str(round(F1[-1], 4)))


if __name__ == '__main__':
    main()


