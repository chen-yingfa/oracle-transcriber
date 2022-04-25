import json

import numpy as np
from PIL import Image


def dump_json(data, fname: str):
    json.dump(data, open(fname, 'w', encoding='utf8'), ensure_ascii=False, indent=2)


def load_json(fname: str):
    return json.load(open(fname, 'r', encoding='utf8'))

def get_img_arr(img: Image) -> np.array:
    w, h = img.size
    arr = np.array(img.getdata()).reshape(h, w)
    return arr

def set_img_arr(img: Image, arr: np.array):
    img.putdata(arr.flatten())

def open_img(path: str, bg_color=(255, 255, 255)) -> Image:
    img = Image.open(path)
    if img.mode in {'RGBA', 'LA'}:
        alpha = img.convert('RGBA').getchannel('A')
        bg = Image.new('RGBA', img.size, bg_color + (255,))
        bg.paste(img, mask=alpha)
        return bg.convert('L')
    else:
        return img.convert('L')


def concat_images(imgs: list, hor: bool=True):
    '''
    hor: Whether to concatenate horizontally.
    '''
    # Creating a new image and pasting the images
    mode = imgs[0].mode
    size = list(imgs[0].size)
    dim = 0 if hor else 1
    size[dim] = sum([img.size[dim] for img in imgs])
    
    concatenated = Image.new(mode, size)
    cur_pos = [0, 0]
    for img in imgs:
        concatenated.paste(img, tuple(cur_pos))
        cur_pos[dim] += img.size[dim]
    return concatenated

