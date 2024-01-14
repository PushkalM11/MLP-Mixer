import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate

def get_imagenet1k_data_loaders(data_dir, batch_size, shuffle, device):
    folders = os.listdir(data_dir)
    classes = np.array([int(f[0 : 3]) for f in folders])
    data = []
    labels = []
    for i in range(len(folders)):
        new_dir = data_dir + folders[i]
        files = os.listdir(new_dir)
        for f in files:
            image = np.asarray(cv2.imread(new_dir + "/" + f), dtype = np.float32).copy() / 255.0
            data.append(image)
            labels.append(classes[i])

    data = np.array(data)
    labels = np.array(labels)

    train_input, test_input, train_output, test_output = train_test_split(
                                                                            data, 
                                                                            labels, 
                                                                            test_size = 0.1, 
                                                                            random_state = 100
                                                                        )

    train_loader = DataLoader(list(zip(train_input, train_output)), 
                            batch_size = batch_size, shuffle = shuffle, 
                            collate_fn = lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_loader = DataLoader(list(zip(test_input, test_output)),
                            batch_size = batch_size, shuffle = shuffle, 
                            collate_fn = lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    return train_loader, test_loader