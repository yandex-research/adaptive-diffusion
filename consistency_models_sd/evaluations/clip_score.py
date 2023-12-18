import torch
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import open_clip
import argparse
import pathlib


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, prompt_filepath, transform=None):
        """
        Args:
            dir_path: path to the stored images
            prompt_filepath: path to the corresponding text prompts (csv file)
        """
        self.dir_path = dir_path
        path = pathlib.Path(dir_path)
        self.files = sorted([
            file for ext in IMAGE_EXTENSIONS
            for file in path.glob('*.{}'.format(ext))
        ], key=lambda x: int(os.path.basename(str(x)).split('.')[0]))

        df = pd.read_csv(prompt_filepath)
        self.text_description = df['caption']
        self.transform = transform
        self.tokenizer = open_clip.get_tokenizer('ViT-g-14')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image).squeeze().float()
        text = self.tokenizer(self.text_description[idx]).squeeze()
        return image, text


@torch.no_grad()
def calculate_clip_score_given_paths(
    image_path, 
    csv_path, 
    batch_size=40
):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
    model = model.eval().cuda()
    dataset = TextImageDataset(image_path, csv_path, transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    clip_score = 0.0
    for image, text in tqdm(loader):
        with torch.cuda.amp.autocast():
            image_features = model.encode_image(image.cuda())
            text_features = model.encode_text(text.cuda())

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_score += (image_features * text_features).sum()
        
    clip_score /= len(dataset)
    return clip_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_dir', type=str)
    parser.add_argument('--prompt_filepath', type=str)
    parser.add_argument('--batch_size', type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(16)

    print(f"Calculating CLIP score for {args.sample_dir}")
    clip_score = calculate_clip_score_given_paths(
        args.sample_dir, args.prompt_filepath,
        batch_size=args.batch_size
    )
    print(f"CLIP score: {clip_score:.3}")