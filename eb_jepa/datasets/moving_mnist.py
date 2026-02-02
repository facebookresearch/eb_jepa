import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Use environment variable or fall back to path relative to __file__
# This allows the original data location to be preserved when running from copied code folder
_DEFAULT_DATASETS_DIR = Path(__file__).parent.parent.parent.absolute() / "datasets"
_DATASETS_DIR = Path(os.environ.get("EBJEPA_DSETS", str(_DEFAULT_DATASETS_DIR)))
FILENAME = str(_DATASETS_DIR / "mnist_test_seq.npy")
URL = "https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"


def load_or_download(filename: str, url: str):
    # Ensure datasets directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        print(f"File '{filename}' not found. Downloading from {url}...")
        try:
            import urllib

            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded '{filename}' successfully.")
        except Exception as e:
            print(f"Failed to download '{filename}': {e}")
            raise
    return filename


def return_bbox(img):
    import cv2

    thres = (img.min() + img.max()) / 2
    contours, _ = cv2.findContours(
        (img > thres).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bounding_boxes = []

    # Loop through contours and extract bounding boxes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append((x, y, x + w, y + h))  # (x_min, y_min, x_max, y_max)
    return bounding_boxes


class MovingMNIST(Dataset):
    def __init__(self, split=None):
        """
        Args:
            split (str): train or val

        Returns:
            video: [C, T, H, W] - greyscale video frames
        """
        load_or_download(FILENAME, URL)
        self.data_path = FILENAME
        dataset = np.load(self.data_path)  # (T, N, H, W) [0-255]
        dataset = np.swapaxes(dataset, 0, 1)

        # Split dataset into train/val (9000/1000)
        assert split in ["train", "val"], "Must choose train or val for split"
        rs = np.random.RandomState(2025)
        dataset = rs.permutation(dataset)
        val_data, train_data = dataset[:1000], dataset[1000:]
        data = val_data if split == "val" else train_data
        # flatten sequences temporally by a factor of 2 to operate on shorter sequences with less memory
        self.data = np.reshape(
            data, [data.shape[0] * 2, data.shape[1] // 2, data.shape[2], data.shape[3]]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = torch.from_numpy(self.data[idx]).unsqueeze(0).float() / 255.0
        return {"video": frames}


class MovingMNISTDet(MovingMNIST):
    def __init__(self, transform=None, split=None, map_size=8):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
            split (str): train or val
            map_size (int): size of map to predict positions over

        Returns:
            video: [C, T, H, W] - greyscale video frames
            digit_location: [T, map_size, map_size] - Coarse binary heatmap for digit locations
        """
        super().__init__(split)
        self.transform = transform

        # Precompute digit locations for all entries
        N, T = self.data.shape[:2]
        self.digit_locations = torch.zeros((N, T, map_size, map_size))
        for idx, frames in enumerate(self.data):
            for t in range(T):
                boxes = return_bbox(frames[t])
                for x1, y1, x2, y2 in boxes:
                    x, y = (x1 + x2) / 2, (y1 + y2) / 2
                    px = int(x / frames.shape[-1] * map_size)
                    py = int(y / frames.shape[-2] * map_size)
                    self.digit_locations[idx, t, py, px] = 1

    def __getitem__(self, idx):
        instance = super().__getitem__(idx)
        digit_locations = self.digit_locations[idx]  # aligned with video
        instance.update({"digit_location": digit_locations})
        return instance


if __name__ == "__main__":
    dset = MovingMNIST(split="val")
    instance = dset[10]
    print(f"{instance['video'].shape = }")

    dset = MovingMNISTDet(split="val")
    instance = dset[10]
    print(f"{instance['video'].shape = }")
    print(f"{instance['digit_location'].shape = }")
