import os
import math
import random

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import blobfile as bf

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def load_echo_data(resolution,
                     batch_size,
                     **kwargs):
    dataset = EchoDataset(resolution=resolution, num_classes=4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
    while True:
        yield from loader


class EchoDataset(Dataset):

    def __init__(self, resolution, num_classes, num_images=100):
        super().__init__()
        self.path = "../../data/camus_cityscape_format/train/"
        self.train_images = _list_image_files_recursively(os.path.join(self.path, "images"))
        self.masks_image = _list_image_files_recursively(os.path.join(self.path, "seg_maps"))
        # sort by name 
        self.train_images.sort()
        self.masks_image.sort()
        assert len(self.train_images) == len(self.masks_image)
        self.data = list(zip(self.train_images, self.masks_image))
        self.resolution = resolution
        self.num_classes = num_classes
        self.num_images = len(self.data)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image_path, mask_path = self.data[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        # resize to resolution
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)
        # convert to numpy
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.int32) 
        # convert to 3 channels
        image = np.stack([image, image, image], axis=-1)
        # convert to one-hot
        # mask = np.eye(self.num_classes)[mask.astype(np.int32)]
        # convert to channels first
        image = np.transpose(image, (2, 0, 1))
        return image, {'label': mask[None,...]}
    
if __name__ == "__main__":
    import matplotlib
    import imageio

    dataset = EchoDataset(resolution=256, num_classes=4)
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1]['label'].shape)
    label = dataset[0][1]['label']
    label = np.argmax(label, axis=0)/3
    label = matplotlib.cm.get_cmap('viridis')(label)
    imageio.imwrite("label.png", label[:, :, :3])