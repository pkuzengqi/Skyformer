# import sys
# import numpy as np
# sys.path.append("..")
import pickle
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import traceback
AUTOTUNE = tf.data.experimental.AUTOTUNE


batch_size = 1
num_classes = 10
vocab_size = 256
input_shape = (1, 32, 32, 1)
normalize = False


def decode(x):
    decoded = {
        'inputs':
            tf.cast(tf.image.rgb_to_grayscale(x['image']), dtype=tf.int32),
        'targets':
            x['label']
    }
    if normalize:
        decoded['inputs'] = decoded['inputs'] / 255
    return decoded



test_dataset = tfds.load('cifar10', split='test')
test_dataset = test_dataset.map(decode, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

val_dataset = tfds.load('cifar10', split='train[90%:]')
val_dataset = val_dataset.map(decode, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

train_dataset = tfds.load('cifar10', split='train[:90%]') # 45000
train_dataset = train_dataset.map(decode, num_parallel_calls=AUTOTUNE) # 45000
# print(train_dataset.cardinality().numpy())
train_dataset = train_dataset.repeat() # -1
# print(train_dataset.cardinality().numpy())
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.shuffle(
    buffer_size=256, reshuffle_each_iteration=True)



mapping = {"test":test_dataset, "dev": val_dataset, "train":train_dataset}

for component in mapping:
    print(component)
    print(mapping[component].cardinality().numpy())

    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        try:
            ds_list.append({
                "input_ids_0":inst["inputs"].numpy()[0].reshape(-1),
                "label":inst["targets"].numpy()[0]
            })
        except:
            print(traceback.format_exc())

        if idx % 100 == 0:
            print(f"{idx}\t\t", end = "\r")

    print(f"dump ../data/lra_processed/lra-image.{component}.pickle")
    with open(f"../data/lra_processed/lra-image.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
