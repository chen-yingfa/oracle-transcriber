from pathlib import Path

DATA_DIR = '../data/rubbing-transcription_2022-04-13/images'

def main():
    path = Path(DATA_DIR)
    # Return a list of (rubbing_file, transcript_file)
    total = 0
    ans = 0
    for glyph_dir in sorted(path.glob('*')):
        img_files = sorted(glyph_dir.glob('*'))
        img_files = [x for x in img_files if x.name != '.DS_Store']
        c = len(img_files)
        total += c
        ans += c * (c - 1)
    print(ans / total)

if __name__ == '__main__':
    main()
