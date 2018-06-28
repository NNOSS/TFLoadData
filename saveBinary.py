import numpy as np
import glob
import loadFeedDict
def write_data(train, train_file_name, test, test_file_name, max_size = 10000):
    i = 0
    while i < len(test):
        append_binary_file(train_file_name,train[i:i+max_size].tobytes())
        append_binary_file(test_file_name,test[i:i+max_size].tobytes())
        i += max_size

def append_binary_file(file_name, bytes_):
    with open(file_name,"ab") as f:
        f.write(bytes_)

def read_binary_file(file_name, dtype):
    with open(file_name,"rb") as f:
        bytes_ = bytearray(f.read())
        print(bytes_)
        print(np.frombuffer(bytes_, dtype=dtype))

def gen_jpg_to_bin(gen, train_file_name, test_file_name):
    train, test = next(gen, None)
    while train is not None:
        append_binary_file(train_file_name,train.tobytes())
        append_binary_file(test_file_name,test.tobytes())
        train, test = next(gen, None)

if __name__ == "__main__":
    gen = loadFeedDict.FromJPEG.get_batches(batch_size, folder, IMAGE_WIDTH,IMAGE_HEIGHT)
