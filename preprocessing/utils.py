import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


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


def process_transcript(img: Image, 
                       bin_threshold: float=96, 
                       crop_black_edges: bool=True) -> Image:
    def get_binary_arr(img: Image) -> np.array:
        # Convert into solid black or white
        arr = get_img_arr(img)
        arr[arr >= bin_threshold] = 255
        arr[arr < bin_threshold] = 0
        return arr

    def binarize(img: Image) -> Image:
        # Inplace
        arr = get_binary_arr(img)
        set_img_arr(img, arr)

    def add_stroke_padding(arr: np.array) -> np.array:
        def in_bounds(i, j):
            return 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]
        radius = 4
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] == 255:
                    # Check if a nearby pixel is black
                    for di in range(1 - radius, radius):
                        for dj in range(1 - radius, radius):
                            x = i + di
                            y = j + dj
                            if in_bounds(x, y) and arr[x, y] == 0:
                                arr[x, y] = 192
        return arr
    
    img = ImageOps.invert(img)
    binarize(img)
    
    if crop_black_edges:
        img = img.crop(img.getbbox())
    return img


def split_image(img: Image) -> tuple:
    w, h = img.size
    split = w // 2
    left = img.crop((0, 0, split, h))
    right = img.crop((split, 0, w, h))
    return left, right


def pad(img: Image, size: int, value=0) -> Image:
    '''Pad an image equally in all directions.'''
    w, h = img.size
    padded = Image.new('L', (w + 2 * size, h + 2 * size), value)
    padded.paste(img, (size, size))
    return padded


def resize_and_pad(img: Image, shape: tuple, pad=0) -> Image:
    '''
    Resize maximally while keeping aspect ratio, then pad up to `shape`.
    shape: (width, height)
    '''
    w, h = img.size
    # Resize
    scale = (shape[0] / w, shape[1] / h)
    min_scale = min(scale)
    resized = img.resize((int(w * min_scale), int(h * min_scale)))
    # resized.save('resized.png')
    
    # Pad
    padded = Image.new('L', shape, pad)
    left = (shape[0] - resized.size[0]) // 2
    top = (shape[1] - resized.size[1]) // 2
    padded.paste(resized, (left, top))
    # padded.save('padded.png')
    
    return padded

def resize_pairs(data_dir: Path, target_shape: tuple=(96, 96)):
    transcript_pairs_dir = data_dir/'pairs'
    dst_dir = data_dir/'oracle2transcription'
    
    count = 0
    for subdir in tqdm(sorted(transcript_pairs_dir.glob('*'))):
        for file in subdir.glob('*'):
            pair = open_img(file)
            left, right = split_image(pair)
            left = resize_and_pad(left, target_shape, pad=0)
            right = resize_and_pad(right, target_shape, pad=0)
            pair = concat_images([left, right], hor=True)
            dst_file = dst_dir / subdir.name / file.name
            dst_file.parent.mkdir(exist_ok=True, parents=True)
            pair.save(dst_file)
            count += 1
    print('Total number of files:', count)


if __name__ == '__main__':
    name = 'temp'
    data_dir = Path('data', name)

    resize_pairs(data_dir)
