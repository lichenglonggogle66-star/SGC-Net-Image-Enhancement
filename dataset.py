import os
from PIL import Image
from torch.utils.data import Dataset

class MIT5K(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = os.listdir(os.path.join(root, "input"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        input_img = Image.open(os.path.join(self.root, "input", name)).convert("RGB")
        gt_img = Image.open(os.path.join(self.root, "gt", name)).convert("RGB")
        if self.transform:
            input_img = self.transform(input_img)
            gt_img = self.transform(gt_img)
        return input_img, gt_img, name
