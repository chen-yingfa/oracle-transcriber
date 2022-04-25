from pathlib import Path

data_dir = '../data/rubbing-transcription_2022-04-13/images'

data_path = Path(data_dir)

total = 0
r = 0

for glyph_dir in data_path.glob('*'):
    cnt = len(list(glyph_dir.glob('*')))
    total += cnt
    r += cnt * (cnt - 1)

r /= total
print('Average number of available substitutions per glyph:', r)
