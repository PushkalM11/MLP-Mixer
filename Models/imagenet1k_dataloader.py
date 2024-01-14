import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torch.utils.data import DataLoader

def get_imagenet_loaders(data_dir, image_size = 224, test_size = 0.2, batch_size = 32, shuffle = True, device = torch.device("cuda")):

    resize = torchvision.transforms.Resize((image_size, image_size), antialias = True)
    def get_image(batch_data):
        images_path, labels = [], []
        for data in batch_data:
            images_path.append(data[0])
            labels.append(data[1])
        data = []
        for img_path in images_path:
            image = cv2.imread(img_path)
            image = torch.tensor(image).type(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            image = resize(image)
            data.append(image)
        data = torch.cat(data)
        labels = torch.tensor(labels)
        return data.to(device), labels.to(device)

    folders = os.listdir(data_dir)
    classes = np.array([int(f[0 : 3]) for f in folders])
    all_files_path = []
    all_files_classes = []

    for i in range(len(folders)):
        files = os.listdir(data_dir + folders[i])
        all_files_path += [data_dir + folders[i] + "/" + f for f in files]
        all_files_classes += [classes[i] for f in files]

    data = all_files_path
    labels = all_files_classes
    train_input, test_input, train_output, test_output = train_test_split(data, labels, test_size = test_size, random_state = 100)

    train_loader = DataLoader(list(zip(train_input, train_output)), 
                            batch_size = batch_size, shuffle = shuffle, collate_fn = get_image)
    test_loader = DataLoader(list(zip(test_input, test_output)),
                            batch_size = batch_size, shuffle = shuffle, collate_fn = get_image)
    
    return train_loader, test_loader