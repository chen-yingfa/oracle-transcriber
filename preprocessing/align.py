import time
from typing import Tuple, List
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

import utils


def normalize(arr):
    avg = np.average(arr)
    std = np.std(arr)
    return (arr - avg) / std


def dist(A, B):
    return np.mean((A - B) ** 2)


def dist_from_center(shape, box_shape, box_pos):
    # print(shape, box_shape, box_pos)
    shape = np.array(shape)
    box_shape = np.array(box_shape)
    box_pos = np.array(box_pos)
    actual = box_pos + box_shape // 2
    target = shape // 2
    return np.sqrt(np.sum((actual - target) ** 2))


def in_bound(pos, shape, bound):
    for i in range(2):
        if pos[i] < 0 or pos[i] + shape[i] >= bound[i]:
            return False
    return True


def align(orig: Image, transcript: Image) -> Tuple[int, int, int, int]:
    '''
    Use block matching to find the position of transcription on original image.

    Return: 
        (match_box, matched_transcript, matched_average_value)
    
    Assuming the aspect ratio is correct and the character is completely within
    the original image, loop different sizes and different positions of the
    transciption.
    
    Complexity: O(min(H, W) * W * H)
    
    Loop different scales such that the shorter side is longer than `min_side`,
    and the longer side is shorter than `long_side`.
    
    Assume that the limiting side is never be smaller than 0.3 of original 
    image.
    '''
    # transcript = preprocess_transcript(transcript)
    
    arr_o = utils.get_img_arr(orig)
    arr_t = utils.get_img_arr(transcript)
    h_o, w_o = arr_o.shape
    h_t, w_t = arr_t.shape
    
    aspect_o = w_o / h_o
    aspect_t = w_t / h_t
    # short_side_ratio = 0.25
    # smallest_shape = w_o * short_side_ratio, h_o * short_side_ratio
    long_side_ratio = 0.7
    smallest_shape = w_o * long_side_ratio, h_o * long_side_ratio
    # print('smallest shape:', smallest_shape)
    # print('transcript shape:', (w_t, h_t))
    # print('orig shape:', (w_o, h_o))

    if w_t > h_t:
        scale_lo = smallest_shape[0] / w_t
        scale_step = 1 / w_t
    else:
        scale_lo = smallest_shape[1] / h_t
        scale_step = 1 / h_t

    if aspect_t > aspect_o:     # Width is the limiting side
        scale_hi = (w_o - 6) / w_t
    else:                       # Height is the limiting side
        scale_hi = (h_o - 6) / h_t
    scale_lo = min(scale_hi - scale_step, scale_lo)
    cur_scale = scale_hi
    
    
    all_dist = []
    
    # Values of the matching block
    match_dist = 9999999999
    match_box = None
    match_arr_t = None
    match_t = None
    
    move_dist = 4
    # Rough grid search, move by 4 pixels each time.
    # Loop different scales
    # print(f'Loop scales: [{scale_lo}, {scale_hi}, {scale_step}]')
    # print(f'Loop iterations:', (scale_hi - scale_lo) / scale_step)
    
    opt_scale = 0.73 * scale_hi  # Optimal scale, from looking at Handa scan pictures
    pos_padding = 0.0
    pos_loop_range = 14  # Length of the sides of the box to search
    
    while cur_scale > scale_lo:
        cur_w = int(cur_scale * w_t)
        cur_h = int(cur_scale * h_t)
        scaled_t = transcript.resize((cur_w, cur_h))
        
        arr_t = np.array(scaled_t.getdata()).reshape(cur_h, cur_w)
        mask = arr_t > 64
    
        # Loop different positions, x: hor, y: ver
        # TODO: Optimize this
        # Just loop around center
        x_lo = (w_o - cur_w - pos_loop_range) // 2
        x_hi = (w_o - cur_w + pos_loop_range) // 2
        y_lo = (h_o - cur_h - pos_loop_range) // 2
        y_hi = (h_o - cur_h + pos_loop_range) // 2
        # x_hi = w_o - cur_w + 1
        # y_hi = h_o - cur_h + 1
        # print(f'Loop positions: [{x_lo}, {x_hi}, {y_lo}, {y_hi}]')
        
        dist_scale = 10000 * abs(cur_scale - opt_scale)
        for x in range(x_lo, x_hi, move_dist):
            for y in range(y_lo, y_hi, move_dist):
                if not in_bound((x, y), (cur_w, cur_h), (w_o, h_o)):
                    continue
                arr_o_cut = arr_o[y:y+cur_h, x:x+cur_w]
                # Since the transcription is solid black background with
                # solid white stroke, we only need to maximize the average value
                # of the part of the original image that overlaps with the
                # strokes.
                # distance = arr_o_cut[mask].mean()
                distance = dist(arr_o_cut, arr_t)
                # distance_from_center = dist_from_center(
                #     (w_o, h_o), (cur_w, cur_h), (x, y))
                # distance += 100 * distance_from_center
                # distance += dist_scale
                all_dist.append(distance)
                if distance < match_dist:
                    match_dist = distance
                    match_box = (x, y, x + cur_w, y + cur_h)
                    match_arr_t = arr_t
                    match_t = scaled_t
                    
        cur_scale -= scale_step
        
    # orig.save('orig.png')
    # transcript.save('transcript.png')
    # print(scale_lo, scale_hi, scale_step)
    
    # Fine grid search, move by 1 pixel each time.
    box_w = match_box[2] - match_box[0]
    box_h = match_box[3] - match_box[1]
    rough_box = match_box
    mask = match_arr_t > 64
    
    for dx in range(1 - move_dist, move_dist):
        for dy in range(1 - move_dist, move_dist):
            x = rough_box[0] + dx
            y = rough_box[1] + dy
            arr_o_cut = arr_o[y:y+box_h, x:x+box_w]
            if arr_o_cut.shape != match_arr_t.shape:
                continue
            # distance = arr_o_cut[mask].mean()
            distance = dist(arr_o_cut, match_arr_t)
            all_dist.append(distance)
            if distance < match_dist:
                match_dist = distance
                match_box = (x, y, x + box_w, y + box_h)
            
    # # Plot
    # print("Match:", match_box)
    # plt.plot(all_dist)
    # plt.xlim(0, len(all_dist))
    # plt.grid()
    # plt.savefig('result/dist.png')
    # plt.clf()
    
    # transcript.save('result/transcript_processed.png')
    
    # match_avg_val = (match_avg_val - np.average(all_dist)) / np.std(all_dist)
    # match_avg_val = (match_avg_val - np.average(all_dist))
    
    return match_box, match_t, match_dist


def align_transcript_to_rubbing(rubbing: Image, transcript: Image) -> Image:
    '''Assume that both transcript and rubbing have black background.'''
    match_box, match_t, _= align(rubbing, transcript)
    new_t = Image.new('L', rubbing.size, 0)
    new_t.paste(match_t, (match_box[0], match_box[1]))
    return new_t


def get_best_match(orig_files: List[str], transcript: Image) -> tuple:
    '''
    Try aligning `transcript` on each originalt image, return the best match.
    '''
    best_box = None
    best_t = None
    best_avg_val = None
    best_idx = None
    # transcript.save('transcript.png')
    for i, file_o in enumerate(orig_files):
        orig = utils.open_img(file_o)
        match_box, match_t, match_avg_val = align(orig, transcript)
        
        # orig.save(f'orig_{i}.png')
        # print(i, 'Distance:', match_avg_val)
        
        if best_box is None or match_avg_val < best_avg_val:
            best_box = match_box
            best_t = match_t
            best_avg_val = match_avg_val
            best_idx = i
        # if i == 4:
        #     exit()
    return best_idx, best_box, best_t, best_avg_val

    
def get_best_aligned_transcript(orig_files: list, transcript: Image) -> Image:
    best_idx, match_box, match_t, _ = get_best_match(orig_files, transcript)
    orig = utils.open_img(orig_files[best_idx])
    w_o, h_o = orig.size
    
    # Concatenate the two images
    arr_t = utils.get_img_arr(match_t)
    new_arr_t = np.zeros((h_o, w_o))
    new_arr_t[match_box[1]:match_box[3], match_box[0]:match_box[2]] = arr_t
    
    new_t = Image.new('L', (w_o, h_o))
    new_t.putdata(list(new_arr_t.flatten()))
    
    # # Draw match box
    # draw = ImageDraw.Draw(orig)
    # draw.rectangle((match_box[0], match_box[1], match_box[2], match_box[3]), outline='gray')
    # orig.save('result/match.png')
    return best_idx, new_t

    
def get_aligned_transcript(orig: Image, transcript: Image) -> Image:
    w_o, h_o = orig.size
    match_box, match_t, _ = align(orig, transcript)
    
    # Concatenate the two images
    arr_t = utils.get_img_arr(match_t)
    new_arr_t = np.zeros((h_o, w_o))
    new_arr_t[match_box[1]:match_box[3], match_box[0]:match_box[2]] = arr_t
    
    new_t = Image.new('L', (w_o, h_o))
    new_t.putdata(list(new_arr_t.flatten()))
    
    # # Draw match box
    # draw = ImageDraw.Draw(orig)
    # draw.rectangle((match_box[0], match_box[1], match_box[2], match_box[3]), outline='gray')
    # orig.save('result/match.png')
    return new_t


def get_aligned_pair_image(orig_files: List[str], transcript: Image) -> Image:
    '''
    Given a transcription and multiple original images, find the best match,
    return them concatenated horizontally.
    '''
    # transcript = utils.open_img(transcript_file)
    idx, aligned_t = get_best_aligned_transcript(orig_files, transcript)
    
    orig = utils.open_img(orig_files[idx])
    joined = utils.concat_images([orig, aligned_t], hor=True)
    return joined


if __name__ == '__main__':
    # transcript_path = '../data/ocr/00037/甲骨文字编（上）_cut_001_3_0_1_人_00037(A7).png'
    transcript_path = Path('../data/transcript_processed/00001/甲骨文字编（上）_cut_001_2_2_1_人_00001(A7).png')
    orig_path = 'H00001-1-5.png'

    orig = utils.open_img(orig_path)
    transcript = utils.open_img(transcript_path)
    transcript.save('result/transcript.png')
    orig.save('result/orig.png')
    
    start_time = time.time()
    aligned = get_aligned_transcript(orig, transcript)
    time_elapsed = time.time() - start_time
    joined = utils.concat_images([orig, aligned], hor=True)
    joined.save('result/joined.png')
    
    print('Time elapsed: {:.2f}s'.format(time_elapsed))
