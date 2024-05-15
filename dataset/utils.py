import numpy as np
import cv2
import pickle
import os


def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save(batch, data, label, filename, save_dir):
    with open('E:\\ImageClassification_CIFAR10\data\data\CIFAR10\\test.txt', 'a') as f:
        for i, (image_data, image_label, image_filename) in enumerate(zip(data, label, filename)):
            image = image_data.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image_name = str(batch) +  str(image_label) + "{:05d}".format(i)
            cv2.imwrite(os.path.join(save_dir, image_name+'.png'), image)

            line_to_write = image_name + ' ' + str(image_label) + '\n'
            f.write(line_to_write)
 
save_dir = "data\\CIFAR10\\test"
base_path = "E:\\ImageClassification_CIFAR10\\data\\cifar-10-batches-py"
batch_files = [
    # "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    "test_batch"]

for i, batch_file in enumerate(batch_files):
    data_batch = unpickle(os.path.join(base_path, batch_file))
    data = data_batch[b'data']
    label = data_batch[b'labels']
    filename = data_batch[b'filenames']

    save(i+1, data, label, filename, save_dir)
    print(f"{batch_file} images saved to {save_dir}")