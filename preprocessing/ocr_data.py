from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

from handa import get_all_book_names


# PATH_OCR_DATA = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res/ocr_char'
# PATH_OCR_DATA = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png/ocr_char'
# PATH_OCR_DATA = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_37/ocr_char'
# PATH_OCR_DATA = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_392/ocr_char'
# PATH_OCR_DATA = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_3223/ocr_char'
# PATH_OCR_DATA = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_323/ocr_char'
PATH_OCR_DATA = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_04011/ocr_char'


# For checking whether book name is valid
all_book_names = None
hanzi_dict = None


def init_all_book_names():
    print('Getting all book names')
    global all_book_names
    all_book_names = get_all_book_names()
    print(f'Got {len(all_book_names)} book names')


def init_hanzi_dict():
    global hanzi_dict

    def load_hanzi_dict(filename: str) -> dict:
        '''Return mapping from index to Chinese char'''
        d = {}
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line[0] == '【':
                    continue
                line = line.split('-')
                if len(line[0]) != 1:
                    continue
                assert len(line[1]) == 4, f'line[1]: {line[1]}'
                hanzi = line[0]
                index = line[1]
                d[index] = hanzi
        return d

    hanzi_dict = load_hanzi_dict('./hanzi_dict.txt')


def get_book_name(filename: str) -> str:
    book_name = filename[:-4].split('_')[-1]
    if '(' in book_name:
        idx = book_name.index('(')
        book_name = book_name[:idx]
    
    puncs = '\'\".,:;()?!（）丶、，。：；？！$¥￥~<>\{\}[]《》【】「］=+-*/#@%^&→↑×° 　＇“”‘’一—－·'
    puncs = set(puncs)
    i = 0
    while i < len(book_name) and book_name[i] in puncs:
        i += 1
    book_name = book_name[i:]
    
    return book_name


def get_hanzi(filename: str) -> str:
    if hanzi_dict is None:
        init_hanzi_dict()
    
    # return filename.split('_')[-2]
    index = filename.split('_')[-3]
    if index in hanzi_dict:
        return hanzi_dict[index]
    else:
        return filename.split('_')[-2]


def get_all_images(src_dir: Path, dst_dir: Path):
    '''
    Get images from `src_dir` and store in `dst_dir`. 
    
    Images are grouped in sub-folders by book name, skips files whose (parsed) 
    book name is empty.
    '''
    print('Getting all images')
    print(f'From {src_dir} to {dst_dir}')
    dst_dir.mkdir(exist_ok=True)
    
    count = 0
    subdirs = sorted(src_dir.glob('*'))
    
    for subdir in tqdm(subdirs):
        for file in subdir.glob('*'):
            book_name = get_book_name(file.name)
            
            if len(book_name) == 0:
                continue
            
            dst_subdir = dst_dir / book_name
            dst_subdir.mkdir(exist_ok=True)
            dst_file = dst_subdir / file.name
            copyfile(file, dst_file)
            count += 1
    print(f'Got {count} images')


def is_valid_book_name(name: str) -> bool:
    # return name in handa.book_dict
    
    return 'H' + name in all_book_names
    
    # if name[-1] in '正反':
    #     name = name[:-1]
    # if len(name) != 5 or not name.isdigit():
    #     return False
    # return True


def is_valid_file_name(name: str) -> bool:
    hanzi = get_hanzi(name)
    if len(hanzi) != 1 or hanzi.isdigit():
        return False
    return True


def get_valid_images(src_dir: Path, dst_dir: Path):
    '''
    Get OCR images from `src_dir` and save only valid ones to `dst_dir`.
    
    Requirements:
    - Book name must be 5 digits with an optional '正' or '反' suffix.
    - Have valid Chinese character label in its file name.
    '''
    print('Getting valid images')
    dst_dir.mkdir(exist_ok=True)
    
    count = 0
    for subdir in tqdm(sorted(src_dir.glob('*'))):
        if not is_valid_book_name(subdir.name):
            continue
        for file in subdir.glob('*'):
            if not is_valid_file_name(file.name):
                continue
            dst_file = dst_dir / subdir.name / file.name
            dst_file.parent.mkdir(exist_ok=True)
            copyfile(file, dst_file)
            count += 1
    print(f'Got {count} images')


def get_images(dst_dir: Path):
    src_dir =  Path(PATH_OCR_DATA)
    dst_dir = Path('data/processed0')
    raw_dst_dir = dst_dir/'ocr_raw'
    # Should take about less than a min in total.
    get_all_images(src_dir, raw_dst_dir)
    get_valid_images(raw_dst_dir, dst_dir)


if __name__ == '__main__':
    src_dir =  Path(PATH_OCR_DATA)
    dst_dir = Path('data')
    raw_dst_dir = dst_dir/'ocr_raw'
    final_dst_dir = dst_dir/'ocr'
    raw_dst_dir.mkdir(exist_ok=True, parents=True)
    final_dst_dir.mkdir(exist_ok=True, parents=True)
    # Should take about less than a min in total.
    get_all_images(src_dir, raw_dst_dir)
    get_valid_images(raw_dst_dir, final_dst_dir)
