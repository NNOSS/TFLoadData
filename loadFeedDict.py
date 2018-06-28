
#Author: Nick Steelman
#Date: 6/28/18
#gns126@gmail.com
#cleanestmink.com
import cPickle
import numpy as np
import os
from PIL import Image
from glob import glob
class CIFAR:
    data_path = '/tmp/cifar10_data'
    data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    @staticmethod
    def pickle_generator(batch_size, file_name):
        data_dict = unpickle(file_name)
        data = data_dict['data']
        labels = data_dict['fine_labels']
        for i in range(0,len(labels),batch_size):
            j = min(i+batch_size,len(labels))
            batch = data[i:j]
            batch = np.reshape(batch, [batch_size, INPUT_SHAPE[2], INPUT_SHAPE[0], INPUT_SHAPE[1]])
            batch = np.transpose(batch, (0, 2, 3, 1))
            yield batch, labels[i:j]

    @staticmethod
    def unpickle(file_name):
        with open(file_name, 'rb') as fo:
            data_dict = cPickle.load(fo)
        return data_dict

    @staticmethod
    def get_class_names(file_name):
        meta_dict = unpickle(file_name)
        return meta_dict['fine_label_names']

    @staticmethod
    def maybe_download_and_extract():
        """Download and extract the tarball from Alex's website."""
        dest_directory = data_path
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        DATA_URL = data_url
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
        if not os.path.exists(extracted_dir_path):
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

class FromJPEG:
    DEBUG = False
    # Image configuration
    data_dir = './Models/data'
    @staticmethod
    def get_image(image_path, width, height, mode, box = None):
        """
        Read image from image_path
        """
        image = Image.open(image_path)
        if image.size != (width, height):
            # Remove most pixels that aren't part of a face
            if box is not None:
                face_width = box[0]
                face_height = box[1]
                j = (image.size[0] - face_width) // 2
                i = (image.size[1] - face_height) // 2
                if i < 0 and j < 0:
                    image = image.resize([face_width, face_height], Image.BILINEAR)
                elif i < 0:
                    image = image.resize([image.size[0], face_height], Image.BILINEAR)
                elif j < 0:
                    image = image.resize([face_width, image.size[1]], Image.BILINEAR)
                image = image.crop([j, i, j + face_width, i + face_height])
                image = image.resize([width, height], Image.BILINEAR)

            else:
                face_width = width
                face_height = height
                j = (image.size[0] - face_width) // 2
                i = (image.size[1] - face_height) // 2
                if i < 0 and j < 0:
                    image = image.resize([face_width, face_height], Image.BILINEAR)
                elif i < 0:
                    image = image.resize([image.size[0], face_height], Image.BILINEAR)
                elif j < 0:
                    image = image.resize([face_width, image.size[1]], Image.BILINEAR)
                image = image.crop([j, i, j + face_width, i + face_height])
        return np.array(image.convert(mode))
    @staticmethod
    def get_batch(image_files, width, height, box = None, mode='RGB'):
        """
        Get a single image
        """
        # print('get file')
        data_batch = np.array(
            [get_image(sample_file, width, height, mode, box=box) for sample_file in image_files]).astype(np.float32)
        # Make sure the images are in 4 dimensions
        if len(data_batch.shape) < 4:
            data_batch = data_batch.reshape(data_batch.shape + (1,))

        return data_batch
    @staticmethod
    def get_batches(batch_size,folder,IMAGE_WIDTH,IMAGE_HEIGHT, box = None):
        """
        Generate batches
        """
        # print('start get_batches')
        current_index = 0
        data_files = glob(folder)
        shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, 3
        #TODO
        labels = None
        # print(shape[0])
        while current_index + batch_size <= shape[0]:
            data_batch = get_batch(
                data_files[current_index:current_index + batch_size],
                *shape[1:3], box=box)
            labels_batch = labels[current_index:current_index + batch_size]
            # print('got files')
            current_index += batch_size
            yield data_batch, labels
