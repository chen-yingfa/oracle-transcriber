from pathlib import Path
import random
from tqdm import tqdm
from shutil import copyfile

def copy_to(files, indices, dst_dir):
    print('Copying files to', dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)
    for idx in tqdm(indices):
        for i in range(2):
            file = files[2*idx + i]
            copyfile(file, dst_dir / file.name)
        
src_dir = Path('220413/individuals_aligned')
dst_dir = Path('220413/individuals_aligned_90')

files = []
for subdir in sorted(src_dir.glob('*')):
    for file in sorted(subdir.glob('*')):
        if file.name.endswith('.png'):
            files.append(file)

files = sorted(files)
cnt = len(files) // 2
indices = list(range(cnt))
random.shuffle(indices)

dev_cnt = int(0.05 * cnt)
train_cnt = cnt - 2*dev_cnt
print(f'train: {train_cnt}')
print(f'dev: {dev_cnt}')
print(f'test: {dev_cnt}')

train_indices = indices[:train_cnt]
dev_indices = indices[train_cnt:train_cnt + dev_cnt]
test_indices = indices[-dev_cnt:]

# exit()
copy_to(files, dev_indices, dst_dir / 'dev')
copy_to(files, test_indices, dst_dir / 'test')
copy_to(files, train_indices, dst_dir / 'train')