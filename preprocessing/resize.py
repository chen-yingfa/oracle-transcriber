from pathlib import Path

from tqdm import tqdm
from PIL import Image

import utils


def split_image(img: Image) -> tuple:
    w, h = img.size
    split = w // 2
    left = img.crop((0, 0, split, h))
    right = img.crop((split, 0, w, h))
    return left, right


def resize(img: Image, shape: tuple, pad=0) -> Image:
    '''
    Resize while keeping aspect ratio.
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
            pair = utils.open_img(file)
            left, right = split_image(pair)
            left = resize(left, target_shape, pad=0)
            right = resize(right, target_shape, pad=0)
            pair = utils.concat_images([left, right], hor=True)
            dst_file = dst_dir / subdir.name / file.name
            dst_file.parent.mkdir(exist_ok=True, parents=True)
            pair.save(dst_file)
            count += 1
    print('Total number of files:', count)


if __name__ == '__main__':
    name = 'temp'
    data_dir = Path('data', name)

    resize_pairs(data_dir)
