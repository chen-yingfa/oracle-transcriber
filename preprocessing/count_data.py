'''Script for counting examples after preprocessing'''

from pathlib import Path

name = '7104'
data_dir = Path('data', name)

'''
Steps:
1. ocr
3. transcript_pairs
4. oracle2transcription
'''

ocr_raw_dir = data_dir/'ocr_raw'
transcript_dir = data_dir/'ocr'
aligned_dir = data_dir/'pairs'
resized_dir = data_dir/'oracle2transcription'


def count_files(dir: Path) -> int:
    count = 0
    for subdir in dir.glob('*'):
        count += len(list(subdir.glob('*')))
    return count

dirs = [
    # ocr_raw_dir,
    # transcript_dir,
    aligned_dir,
    resized_dir,
]

for dir in dirs:
    print(f'{dir}: {count_files(dir)}')
