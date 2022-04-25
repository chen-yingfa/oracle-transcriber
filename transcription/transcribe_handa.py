from pathlib import Path

from PIL import Image
from tqdm import tqdm

from transcribe import transcribe, get_model


src_path = Path('/data/private/songchenyang/hanzi_filter/handa/H/characters')
dst_path = Path('../data/handa/H/characters')
model = get_model()


def open_img(file):
    img = Image.open(file)
    img.convert('L')
    return img
    
dst_path.mkdir(exist_ok=True, parents=True)

total = 319200  # Pre-computed
for file in tqdm(src_path.glob('*'), total=total):
    dst_file = dst_path/file.name
    img = open_img(file)
    img = transcribe(img, model)
    img.save(dst_file)