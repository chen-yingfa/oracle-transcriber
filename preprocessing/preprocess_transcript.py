from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import utils


def process_transcript(img: Image, bin_threshold: float=96) -> Image:
    # Convert into solid black or white
    def get_binary_arr(img: Image) -> np.array:
        arr = utils.get_img_arr(img)
        # print(arr[60:100, 16:32])
        arr[arr >= bin_threshold] = 255
        arr[arr < bin_threshold] = 0
        return arr
        
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
                        
        # print(arr[60:100, 16:32])
        return arr
    img = ImageOps.invert(img)
    arr = get_binary_arr(img)
    utils.set_img_arr(img, arr)
    
    # Remove padding
    bbox = img.getbbox()
    img = img.crop(bbox)
    return img


def preprocess(file) -> Image:
    img = utils.open_img(file)
    return process_transcript(img)


def main():
    data_dir = Path('data')
    dst_dir = data_dir/'transcript_processed'
    # padded_dir = Path('../data/transcript_padded')
    # padded_dir.mkdir(exist_ok=True)
    dst_dir.mkdir(exist_ok=True)
    src_dir = data_dir/'ocr'
    # Preprocess transcript images, and save.
    for src_subdir in tqdm(sorted(src_dir.glob('*'))):
        dst_subdir = dst_dir / src_subdir.name
        dst_subdir.mkdir(exist_ok=True)
        for file in sorted(src_subdir.glob('*')):
            img = utils.open_img(file)
            dst_file = dst_subdir / file.name
            processed = process_transcript(img)
            processed.save(dst_file)
    

if __name__ == '__main__':
    main()