from pathlib import Path
from shutil import copyfile
import random

from tqdm import tqdm

src_dir = Path('../preprocessing/data/220413/individuals')
dst_dir = Path('datasets/220413/individuals')

def get_files(dir: Path):
    files = []
    for subdir in sorted(dir.glob('*')):
        for file in sorted(subdir.glob('*')):
            if file.name.endswith('r.png'):
                files.append(file)
    return files

print(f"Getting files from {src_dir}")

src_files = get_files(src_dir)
random.seed(0)
random.shuffle(src_files)
print(f'Found {len(src_files)} files')

# Split into train, dev, test
test_size = int(0.02 * len(src_files))
dev_size = int(0.02 * len(src_files))

test_files = src_files[:test_size]
dev_files = src_files[test_size:test_size + dev_size]
train_files = src_files[test_size + dev_size:]
print(f'# train examples: {len(train_files)}')
print(f'# dev examples: {len(dev_files)}')
print(f'# test examples: {len(test_files)}')
train_dir = dst_dir / 'train'
dev_dir = dst_dir / 'val'
test_dir = dst_dir / 'test'
train_dir.mkdir(exist_ok=True, parents=True)
dev_dir.mkdir(exist_ok=True, parents=True)
test_dir.mkdir(exist_ok=True, parents=True)

def copyfiles(files, dst_dir):
    for file in tqdm(files):
        dst_file_r = dst_dir / file.name
        copyfile(file, dst_file_r)
        # Copy transcription file
        file_t = file.parent / file.name.replace('r.png', 't.png')
        dst_file_t = dst_dir / file_t.name
        copyfile(file_t, dst_file_t)

print(f'Copying to {test_dir}')
copyfiles(test_files, test_dir)
print(f'Copying to {dev_dir}')
copyfiles(dev_files, dev_dir)
print(f'Copying to {train_dir}')
copyfiles(train_files, train_dir)
