# !export KAGGLE_USERNAME=iamtars
# !export KAGGLE_KEY=e10e91505fe6ab8a093cc0f348efb35e
# ! mkdir ~/.kaggle
# ! cp kaggle.json ~/.kaggle/
# ! chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d adityajn105/flickr8k
# import subprocess
# zip_file_path = "/content/flickr8k.zip"
# extract_path = "/content/flickr8k"
# subprocess.run(["unzip", zip_file_path, "-d", extract_path])

from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class Dataset(Dataset):
    def __init__(self):
        self.imgspath = os.listdir("/content/flickr8k/Images")
        with open("/content/flickr8k/captions.txt", 'r') as file:
          lines = file.readlines()
        self.captions = {}
        lines = lines[1:]
        for line in lines:
          if ',' in line:
            image, caption = line.strip().split(",", 1)
            self.captions[image] = caption
    def __len__(self):
        return len(self.imgspath)

    def __getitem__(self, idx):
        img = os.path.join("/content/flickr8k/Images",self.imgspath[idx])
        label = self.captions[self.imgspath[idx]]
        img = Image.open(img)
        img = transform(img)
        return img, (label)
batch_size = 32
dataset = Dataset()
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)