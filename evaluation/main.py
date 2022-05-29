from pathlib import Path
import json

from dcp import get_transcription_dcp
from tcp import get_transcription_tcp
from mse import get_mse

if __name__ == '__main__':
    test_names = [
        # '220413_replace0_mask1',
        # '220413_replace0.4_mask0',
        # '220413_replace0.8_mask0',
        # '220413_aligned',
        # '220413',
        '220413_aligned_replace0_hmask0.8_smask0',
        '220413_aligned_replace0_hmask0.8_smask0.4',
        '220413_aligned_replace0_hmask0.8_smask0.8',
    ]

    for test_name in test_names:
        print(test_name)
        mse = get_mse(test_name)
        dcp = get_transcription_dcp(test_name)
        tcp = get_transcription_tcp(test_name)
        
        print(f'  MSE: {mse:.4f}')
        print(f'  DCP: {dcp:.2f}')
        print(f'  TCP: {tcp:.2f}')