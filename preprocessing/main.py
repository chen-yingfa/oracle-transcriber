from pathlib import Path

from gen_pairs import gen_pairs_scy
from resize import resize_pairs



if __name__ == '__main__':
    name = '6912'
    data_dir = Path('data', name)
    target_shape = (96, 96)
    
    # print("Generating pairs")
    # gen_pairs_scy(data_dir)
    
    print("Resizing pairs")
    resize_pairs(data_dir, target_shape=target_shape)