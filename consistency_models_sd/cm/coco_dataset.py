import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import csv
import pandas as pd
import cm.dist_util as dist


def imagenet_transform(size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop([size, size]),
        transforms.ToTensor(),
        normalize,])


def adaptive_image_resize(x, h, w):
    if x.size[0] > x.size[1]:
        t = transforms.Resize(h)
    else:
        t = transforms.Resize(w)
    return t(x)


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, subset_name='subset', transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
        sample_dir = os.path.join(root_dir, subset_name)
        self.samples = sorted([os.path.join(sample_dir, fname) for fname in os.listdir(sample_dir) 
                        if fname[-4:] in self.extensions], key=lambda x: x.split('/')[-1].split('.')[0])
        
        self.captions = {}
        with open(os.path.join(root_dir, f"{subset_name}.csv"), newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(spamreader):
                if i == 0:
                    continue
                self.captions[row[1]] = row[2]
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_path = self.samples[idx]
        sample = Image.open(sample_path).convert('RGB')

        if self.transform:
            sample = self.transform(sample)

        return {'image': sample, 'text': self.captions[os.path.basename(sample_path)]}
    

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1
    

def prepare_coco_prompts(path, bs=20, max_cnt=10000):
    df = pd.read_csv(path)
    all_text = list(df['caption'])
    all_text = all_text[:max_cnt]

    num_batches = ((len(all_text) - 1) // (bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()] 

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]
    return rank_batches, rank_batches_index