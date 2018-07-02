import tensorflow as tf
import numpy as np

class MNIST:
    mnist_file_path = '/Data/MNIST_data/'
    @staticmethod
    def return_dataset_train():
        images_file = mnist_file_path + 'train-images-idx3-ubyte'
        labels_file = mnist_file_path + 'train-labels-idx1-ubyte'
        images = tf.data.FixedLengthRecordDataset(
          images_file, 28 * 28, header_bytes=16).map(mnist_decode_image)
        labels = tf.data.FixedLengthRecordDataset(
          labels_file, 1, header_bytes=8).map(mnist_decode_label)
        return tf.data.Dataset.zip((images, labels))
    @staticmethod
    def return_dataset_test():
        images_file = mnist_file_path + 't10k-images-idx3-ubyte'
        labels_file = mnist_file_path + 't10k-labels-idx1-ubyte'
        images = tf.data.FixedLengthRecordDataset(
          images_file, 28 * 28, header_bytes=16).map(mnist_decode_image)
        labels = tf.data.FixedLengthRecordDataset(
          labels_file, 1, header_bytes=8).map(mnist_decode_label)
        return tf.data.Dataset.zip((images, labels))
    @staticmethod
    def mnist_decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [28, 28, 1])
        return image / 255.0
    @staticmethod
    def mnist_decode_label(label):
        label = tf.decode_raw(label, tf.uint8) + 1  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

class ImageNet:
    imagenet_file_path ='/Data/Imagenet/'
    IMAGE_SIZE = 512, 512, 3
    IMAGE_LEN = IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]
    NUM_TRAIN = 5
    NUM_TEST = 1
    NUM_CLASSES = 200
    NUM_BYTES = 1

    # train_images = ['train_images_'+ str(i) for i in range(num_train)]
    # train_labels = ['train_labels_'+ str(i) for i in range(num_train)]
    train_images = ['train_images']
    train_labels = ['train_labels']
    test_images = ['test_images'+ str(i) for i in range(NUM_TEST)]
    test_labels = ['test_labels'+ str(i) for i in range(NUM_TEST)]
    @staticmethod
    def return_dataset_train():
        images_file = [ImageNet.imagenet_file_path + val for val in ImageNet.train_images]
        labels_file = [ImageNet.imagenet_file_path + val for val in ImageNet.train_labels]
        images = tf.data.FixedLengthRecordDataset(
          images_file, ImageNet.IMAGE_LEN * ImageNet.NUM_BYTES).map(ImageNet.decode_image)
        labels = tf.data.FixedLengthRecordDataset(
          labels_file, ImageNet.NUM_CLASSES).map(ImageNet.decode_label)
        return tf.data.Dataset.zip((images, labels))
    @staticmethod
    def return_dataset_test():
        images_file = [mnist_file_path + val for val in train_images]
        labels_file = [mnist_file_path + val for val in train_labels]
        images = tf.data.FixedLengthRecordDataset(
          images_file, IMAGE_LEN * NUM_BYTES).map(decode_image)
        labels = tf.data.FixedLengthRecordDataset(
          labels_file, NUM_CLASSES).map(decode_label)
        return tf.data.Dataset.zip((images, labels))
    @staticmethod
    def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [ImageNet.IMAGE_SIZE[0], ImageNet.IMAGE_SIZE[1],ImageNet.IMAGE_SIZE[2]])
        return image / 255.0
    @staticmethod
    def decode_label(label):
        label = tf.decode_raw(label, tf.int8)# tf.string -> [tf.uint8]
        return tf.to_int32(label)



if __name__ == "__main__":
    DATA_PATH = '/Data/cifar-100-python/'
    TRAINING_NAME = 'train'
    TESTING_NAME = 'test'
    META_NAME = 'meta'
    print(get_class_names(DATA_PATH + META_NAME))
