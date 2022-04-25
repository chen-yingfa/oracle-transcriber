from pathlib import Path

from PIL import Image
from tqdm import tqdm

from transcribe import transcribe, get_model


src_path = Path('../data/hanzi_filter/classes')
dst_path = Path('../data/handa_classes')
model = get_model()


def open_img(file):
    def pad_to_square(img):
        w, h = img.size
        longest = max(w, h)
        padded = Image.new('L', (longest, longest))
        if w < h:
            paste_pos = ((longest - w) // 2, 0)
        else:
            paste_pos = (0, (longest - h) // 2)
        padded.paste(img, paste_pos)
        return padded 

    img = Image.open(file)
    img.convert('L')
    img = pad_to_square(img)
    img = img.resize((96, 96))
    return img
    

for subdir0 in tqdm(sorted(src_path.glob('*'))):
    for subdir1 in sorted(subdir0.glob('*')):
        for file in sorted(subdir1.glob('*')):
            dst_file = dst_path/subdir0.name/subdir1.name/file.name
            img = open_img(file)
            img = transcribe(img, model)
            dst_file.parent.mkdir(exist_ok=True, parents=True)
            img.save(dst_file)

