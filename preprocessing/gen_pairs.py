from pathlib import Path
import json

from tqdm import tqdm
import opencc
from PIL import Image, ImageOps
import numpy as np

import align
from handa import Handa
import ocr_data
import utils
from noisy_transcript_map import get_files


hanzi_dict = None


def init_hanzi_dict():
    global hanzi_dict

    def load_dict(file: str) -> dict:
        d = {}
        for line in open(file, 'r', encoding='utf8'):
            line = line.strip().split()
            if len(line) == 2:
                d[line[0]] = line[1]
        return d

    hanzi_dict = load_dict('map_hanzi.txt')


def process_hanzi(hanzi: str) -> str:
    '''Convert hanzi label to the one used by Handa (HK traditional)'''
    converter = opencc.OpenCC('s2hk.json')
    hanzi = converter.convert(hanzi)
    if hanzi in hanzi_dict:
        return hanzi_dict[hanzi]
    else:
        return hanzi


def gen_pairs(data_dir: Path):
    '''
    Takes 30 min
    '''
    ocr_data_dir = data_dir/'ocr'
    pair_data_dir = data_dir/'pairs'
    
    init_hanzi_dict()
    
    print('Initting handa')
    handa = Handa()
    print(f'Looping OCR files in {ocr_data_dir}')
    print(f'Saving paired images to {pair_data_dir}')
    count = 0
    for subdir in tqdm(sorted(ocr_data_dir.glob('*'))):
        book_name = 'H' + subdir.name
        for transcript_file in sorted(subdir.glob('*')):
            # TODO: Use block matching to find the corresponding original image.
            
            # Currently, just look up using book name and Chinese characters,
            # they uniquely identify an original image.
            hanzi = ocr_data.get_hanzi(transcript_file.name)
            hanzi = process_hanzi(hanzi)
            orig_file = handa.get_file(book_name, hanzi)
            if orig_file is not None:
                # transcript = utils.open_img(transcript_file)
                # transcript = process_transcript(transcript)                
                # paired_img = align.get_aligned_pair_image([orig_file], transcript)
                
                # paired_img = transcript
                
                # dst_file = pair_data_dir / book_name / (hanzi+'.png')
                # dst_file.parent.mkdir(exist_ok=True, parents=True)
                # paired_img.save(dst_file)
                count += 1
            
    print('Number of pairs:', count)
    
    json.dump(
        handa.not_found, 
        open('not_found.json', 'w', encoding='utf8'), 
        ensure_ascii=False, 
        indent=2)


def gen_pairs_scy(output_dir):
    '''
    Use database from Song Chenyang that records the mapping of transcription to
    original noisy images.

    Takes 30 min.
    '''
    # PATH_TRANSCRIPT = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_324/ocr_char'
    PATH_TRANSCRIPT = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_04011/ocr_char'
    PATH_NOISE = '/data/private/songchenyang/hanzi_filter/handa'
    output_dir = output_dir/'pairs'
    
    print('Getting file mapping from database')
    files = get_files()
    files = list(files)  # ~10k elements
    print(f'Got {len(files)} files')
    for noise_file, transcript_file in tqdm(files):
        noise_file = Path(PATH_NOISE, noise_file)
        transcript_file = Path(PATH_TRANSCRIPT, transcript_file)
        
        transcript = utils.open_img(transcript_file)
        transcript = utils.process_transcript(transcript)
        paired_img = align.get_aligned_pair_image([noise_file], transcript)
        
        book_name = ocr_data.get_book_name(transcript_file.name)
        hanzi = ocr_data.get_hanzi(transcript_file.name)
        
        dst_file = output_dir / book_name / (hanzi + '.png')
        dst_file.parent.mkdir(exist_ok=True, parents=True)
        paired_img.save(dst_file)
    print('done')


if __name__ == '__main__':
    name = 'temp'
    data_dir = Path('data', name)
    gen_pairs_scy(data_dir)
    # gen_pairs(data_dir)
