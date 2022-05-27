from pathlib import Path

from PIL import Image
import numpy as np


def img_dist(a, b):
    a = np.array(a.getdata())
    b = np.array(b.getdata())
    return np.average(((a - b) / 255) ** 2)


def mse(test_name):
    gen_dir = Path(f'../oracle-transcriber/transcription/results/{test_name}/test_latest/images')
    fake_bs = [img for img in gen_dir.glob('*fake_B.png')]
    real_bs = [img for img in gen_dir.glob('*real_B.png')]
    
    total_dist = 0
    for fake_path, real_path in zip(fake_bs, real_bs):
        fake = Image.open(fake_path).convert('L')
        real = Image.open(real_path).convert('L')
        total_dist += img_dist(fake, real)
    print(f'Total MSE: {total_dist / len(fake_bs)}')
    
    
for test_name in [
    '220413_replace0_mask1',
    '220413_replace0.4_mask0',
    '220413_replace0.8_mask0',
    '220413_aligned'
    ]:
    print(test_name)
    mse(test_name)
