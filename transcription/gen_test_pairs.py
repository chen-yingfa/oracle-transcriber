from pathlib import Path

from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np

import utils


src_dir = Path('datasets/rubbing_531/raw')
processed_dir = Path('datasets/rubbing_531/processed')
dst_dir = Path('datasets/rubbing_531/pairs/test')

processed_dir.mkdir(exist_ok=True, parents=True)
dst_dir.mkdir(exist_ok=True, parents=True)


def preprocess(image: Image) -> Image:
    def force_white_carvings(img):
        '''Try to turn carvings into white'''
        arr = np.array(img.getdata()).reshape(img.size[1], img.size[0])
        split = 128
        arr[arr > split] = 255
        arr[arr <= split] = 0
        if arr[5:-5, 5:-5].mean() > 128:
            return ImageOps.invert(image)
        return image
    
    def pad_and_resize(img: Image, shape: tuple=(96, 96), pad: int=0) -> Image:
        w, h = img.size
        longest = max(w, h)
        paste_pos = ((longest - w) // 2, (longest - h) // 2)
        new_img = Image.new('L', (longest, longest), color=pad)
        new_img.paste(img, paste_pos)
        new_img = new_img.resize(shape)
        return new_img
    # image = force_white_carvings(image)
    image = pad_and_resize(image)
    return image



print('Generating pairs...')
for file in tqdm(sorted(src_dir.glob('*'))):
    image = utils.open_img(file)
    image = preprocess(image)
    image.save(processed_dir / file.name)
    
    # Concatenate into pairs
    image = utils.concat_images([image, image])
    dst_file = dst_dir / file.name
    image.save(dst_file)
