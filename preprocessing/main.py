from pathlib import Path
import random

from tqdm import tqdm

import utils
import align

DATA_DIR = '../data/rubbing-transcription_2022-04-13/images'


def iter_pair_files(path: Path):
    # Return a list of (rubbing_file, transcript_file)
    for glyph_dir in sorted(path.glob('*')):
        img_files = sorted(glyph_dir.glob('*'))
        img_files = [x for x in img_files if x.name != '.DS_Store']
        assert len(img_files) % 2 == 0, f'There must an even number of images, \
ie. pair of images for each glyph, but {glyph_dir.name} has {len(img_files)} images.'
        for i in range(0, len(img_files), 2):
            yield img_files[i], img_files[i+1]
            
            
def save_pair(transcript, rubbing, dst_glyph_dir, fname, do_align=False):
    
    if align:
        transcript = align.align_transcript_to_rubbing(rubbing, transcript)
    pair = utils.concat_images([rubbing, transcript], hor=True)
    pair_file = dst_glyph_dir / (fname + '.png')
    pair.save(pair_file)


def save_files(transcript, rubbing, dst_glyph_dir, pref, do_align=False):
    fname_t = pref + '_t.png'
    fname_r = pref + '_r.png'
    transcript.save(dst_glyph_dir / fname_t)
    rubbing.save(dst_glyph_dir / fname_r)


def main():
    data_path = Path(DATA_DIR)
    name = '220413'
    dst_dir = Path('data', name, 'individuals')
    target_shape = (96, 96)
    
    print(f'Loading from {data_path}')
    pair_files = list(iter_pair_files(data_path))
    print(f'Found {len(pair_files)} pairs of images.')
    
    print(f'Destination path: {dst_dir}')
    for rubbing_file, transcript_file in tqdm(pair_files):
        pref_t = transcript_file.name[:7]
        pref_r = rubbing_file.name[:7]
        if pref_t != pref_r:
            # Currently, this only occur once.
            continue
        glyph = transcript_file.name[0]
        
        # Preprocess
        pad_transcript = 6
        
        transcript = utils.open_img(transcript_file)
        transcript = utils.process_transcript(transcript)
        transcript = utils.pad(transcript, pad_transcript)
        transcript = utils.resize_and_pad(transcript, target_shape)
        # transcript.save('transcript.png')
        
        rubbing = utils.open_img(rubbing_file)
        rubbing = utils.resize_and_pad(rubbing, target_shape)
        # rubbing.save('rubbing.png')
        
        # Save
        dst_glyph_dir = dst_dir / glyph
        dst_glyph_dir.mkdir(exist_ok=True, parents=True)
        
        save_files(transcript, rubbing, dst_glyph_dir, pref_t, do_align=False)
        # save_pair(transcript, rubbing, dst_glyph_dir, pref_t, do_align=True)
    print('Done processing')

if __name__ == '__main__':
    main()
