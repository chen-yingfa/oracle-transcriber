from pathlib import Path

from gen_pairs import gen_pairs_scy
from resize import resize_pairs

DATA_DIR = '../data/rubbing-transcription_2022-04-13/images'

if __name__ == '__main__':
    name = 'r2t_220413'
    data_dir = Path('data', name)
    target_shape = (96, 96)
    
    data_path = Path(DATA_DIR)
    for glyph_dir in data_path.glob('*'):
        cnt = len(list(glyph_dir.glob('*')))
        print(glyph_dir.name, cnt)