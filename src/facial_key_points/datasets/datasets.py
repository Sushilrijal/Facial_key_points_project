import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from src.facial_key_points.config.config import configuration


class FacialKeyPointsDataset(Dataset):
    def __init__(
        self,
        csv_file_path,
        split="training",
        model_input_size=224,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.csv_file_path = csv_file_path
        self.split = split
        self.model_input_size = model_input_size
        self.df = pd.read_csv(self.csv_file_path)
        # print(self.df)    -->1
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # [r,g,b]
            std=[0.229, 0.224, 0.225],
        )
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img, original_size = self.get_img(idx)
        facial_keypoints = self.get_keypoints(index=idx, original_size=original_size)
        return img, facial_keypoints

    def get_img(self, index):  # to get img from  training folder using the keypoints df
        # print(self.df.iloc[index,0])   # --> 3
        # print(os.path.join(os.getcwd(), 'data', self.split, self.df.iloc[index,0]))   # ---> 4
        img_path = os.path.join(
            os.path.join(os.getcwd(), "data", self.split, self.df.iloc[index, 0])
        )
        img = Image.open(img_path).convert("RGB")
        original_size = img.size
        # print(original_size)    # ---> 5

        # pre-process image  ( normalizing and then converting to tensors)
        img = img.resize((self.model_input_size, self.model_input_size))
        # print(img.size)
        # print(np.asarray(img))
        img = (
            np.asarray(img) / 255.0
        )  # range of pixel value is between 0(black) and 255(white), we normalize it to [0,1]
        # print(img.shape)  ---> (224,224,3)  i.e ( height , width, channels(RGB))

        # but in pytorch image_tensor should nbe represented in standard form ( batch,channel, width,height), to solve it we use permute
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        # print(img.shape)    --> torch.Size([1, 3, 224, 224])
        img = self.normalize(img)
        # print(img)
        return img.to(self.device), original_size

    def get_keypoints(self, index, original_size):
        kp = (
            self.df.iloc[index, 1:].to_numpy().astype(np.float32)
        )  # since kp_x,y is list ,we cant divide it by int . So, we use broadcasting concept of numpy
        kp_x = kp[0::2] / original_size[0]
        kp_y = kp[1::2] / original_size[1]
        kp = np.concatenate([kp_x, kp_y]).astype(np.float32)  # required ip to the model
        # print(type(kp))
        # print(kp)
        return torch.tensor(kp).to(self.device)

    def load_img(self, index):  # to use it for visualization
        img_path = os.path.join(os.getcwd(), "data", self.split, self.df.iloc[index, 0])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.model_input_size, self.model_input_size))
        return np.asarray(img) / 255.0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_data = FacialKeyPointsDataset(
        csv_file_path=configuration["train_data_csv_path"], device=device
    )
    print(training_data[0])
    # test_data = FacialKeyPointsDataset(csv_file_path=r'data/test_frames_keypoints.csv', split='test', device=device)
