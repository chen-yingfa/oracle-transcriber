
'''
Use database from Song Chenyang that records the mapping of transcription to
original noisy images.
'''
from tqdm import tqdm
from noisy_transcript_map import get_files

files = get_files()
files = list(files)

# N, T = zip(*files)
# print(len(set(N)))
# print(len(set(T)))

with open('noise_transcript_map.txt', 'w', encoding='utf8') as f:
    for noise_file, transcript_file in tqdm(files):
        f.write(f'{noise_file} {transcript_file}\n')

print('done')