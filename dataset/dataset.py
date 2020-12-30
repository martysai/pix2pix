import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from .helpers import make_dataset, get_transforms


class MyDataset(Dataset):
    def __init__(self,
                 args,
                 data="facades",
                 mode="train",
                 flip_ratio=0.5,
                 left_to_right=True):
        super(MyDataset, self).__init__()
        self.datapath = os.path.join(data, mode)
        self.paths = sorted(make_dataset(self.datapath))
        self.transforms = get_transforms(args)
        self.flip_ratio = flip_ratio

        # Определяем, с какой стороны
        # исходное изображение, а с какой -- целевое
        self.left_to_right = left_to_right

    def __getitem__(self, index):
        path = self.paths[index]
        im = Image.open(path).convert('RGB')

        w, h = im.size
        w2 = int(w / 2)
        src, trg = None, None
        if self.left_to_right:
            src = im.crop((0, 0, w2, h))
            trg = im.crop((w2, 0, w, h))
        else:
            trg = im.crop((0, 0, w2, h))
            src = im.crop((w2, 0, w, h))

        src = self.transforms(src)
        src = np.array(src)

        trg = self.transforms(trg)
        trg = np.array(trg)

        return {"src": src, "trg": trg, "path": path}

    def __len__(self):
        return len(self.paths)
