from pathlib import Path
from PIL import Image
import torch
import yaml
import math

import torchvision.transforms as T
from torchvision.io import read_video,write_video
import os
import random
import numpy as np
from torchvision.io import write_video
# from kornia.filters import joint_bilateral_blur
from kornia.geometry.transform import remap
from kornia.utils.grid import create_meshgrid
import cv2

def save_video_frames(video_path, img_size=(512,512)):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith('.mov'):
        video = T.functional.rotate(video, -90)
    video_name = Path(video_path).stem
    os.makedirs(f'data/{video_name}', exist_ok=True)
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        image_resized = image.resize((img_size),  resample=Image.Resampling.LANCZOS)
        image_resized.save(f'data/{video_name}/{ind}.png')

def parse_index_spec(spec):
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str) and os.path.isfile(spec):
        with open(spec, "r") as f:
            spec = f.read().strip()
    if not isinstance(spec, str):
        raise ValueError(f"Unsupported index spec type: {type(spec)}")
    tokens = [tok.strip() for tok in spec.replace("\n", ",").split(",") if tok.strip()]
    indices = []
    for token in tokens:
        if "-" in token:
            start, end = token.split("-", 1)
            start = int(start.strip())
            end = int(end.strip())
            step = 1 if end >= start else -1
            indices.extend(list(range(start, end + step, step)))
        else:
            indices.append(int(token))
    return indices

def get_image_paths(data_path, n_frames, view_indices=None):
    view_indices = parse_index_spec(view_indices)
    if view_indices is None:
        view_indices = list(range(n_frames))
    paths = []
    for idx in view_indices:
        path = os.path.join(data_path, "%05d.png" % idx)
        if not os.path.exists(path):
            path = os.path.join(data_path, "%05d.jpg" % idx)
        paths.append(path)
    return paths

def add_dict_to_yaml_file(file_path, key, value):
    data = {}

    # If the file already exists, load its contents into the data dictionary
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

    # Add or update the key-value pair
    data[key] = value

    # Save the data back to the YAML file
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
        
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def load_imgs(data_path, n_frames, device='cuda', pil=False):
    imgs = []
    pils = []
    for i in range(n_frames):
        img_path = os.path.join(data_path, "%05d.jpg" % i)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_path, "%05d.png" % i)
        img_pil = Image.open(img_path)
        pils.append(img_pil)
        img = T.ToTensor()(img_pil).unsqueeze(0)
        imgs.append(img)
    if pil:
        return torch.cat(imgs).to(device), pils
    return torch.cat(imgs).to(device)


def save_video(raw_frames, save_path, fps=10):
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }

    frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
