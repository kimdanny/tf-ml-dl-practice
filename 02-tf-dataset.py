import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# print(tfds.list_builders()) # list all the available datasets

# Loading dataset
dataset = tfds.load('mnist', split='train', shuffle_files=True)
assert isinstance(dataset, tf.data.Dataset)  # tf.data.Dataset is a higher class
# print(dataset)

"""
tfds.load is a thin wrapper around tfds.core.DatasetBuilder 
"""
builder = tfds.builder('mnist')
builder.download_and_prepare()
mnist = builder.as_dataset(split='train', shuffle_files=True)
# print(mnist)  # Get the same thing as above

"""
Iterate over dataset
"""
sample = mnist.take(1) # take a single sample

draw_image = False
for example in sample:
    print(list(example.keys()))
    image = example['image']
    label = example['label']
    # image shape is (28, 28, 1)
    # print("shape of image: {} \t label: {}".format(image.shape, label))
    if draw_image:
        plt.matshow(np.array(image).reshape((28, 28)))
        plt.title(f"Label: {label}")
        plt.show()

"""
Get image as a numpy array
"""
# By using as_supervised=True option, you can get a tuple (features, label) for supervised datasets
datset = tfds.load('mnist', split='train', as_supervised=True)
sample = datset.take(1)

for image, label in tfds.as_numpy(datset):
    print(type(image), type(label), label)


"""
@misc{TFDS,
  title = { {TensorFlow Datasets}, A collection of ready-to-use datasets},
  howpublished = {\url{https://www.tensorflow.org/datasets} },
}
"""
