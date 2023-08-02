import os
import random
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = self.get_AB_paths(self.dir_AB, opt.max_dataset_size)  # get image paths
        self.glyph_to_indices = self.get_glyph_to_indices()
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        
        # Data augmentation
        self.phase = opt.phase
        if opt.phase == 'train':
            self.replace_prob = opt.da_replace_prob
            self.hmask_prob = opt.da_hmask_prob
            self.smask_prob = opt.da_smask_prob
            
            self.transform_random_erasing = transforms.RandomErasing(
                p=self.smask_prob, value='random')
        
            # # Heuristically generated mask files
            # self.hmask_dir = '../data/mask'
            # self.hmask_files = os.listdir(self.hmask_dir)
        
    def get_AB_paths(self, data_dir, max_dataset_size):
        # Load all image paths in a list of path pairs.
        all_paths = sorted(make_dataset(data_dir, max_dataset_size))  # get image paths
        AB_paths = []
        for i in range(0, len(all_paths), 2):
            # The second path is transcription (B)
            AB_paths.append((all_paths[i], all_paths[i+1]))
        return AB_paths
    
    def get_glyph(self, path):
        return path.split('/')[-1][0]
        
    def get_glyph_to_indices(self):
        glyph_to_indices = {}
        for i in range(len(self.AB_paths)):
            glyph = self.get_glyph(self.AB_paths[i][0])
            if glyph not in glyph_to_indices:
                glyph_to_indices[glyph] = []
            glyph_to_indices[glyph].append(i)
        return glyph_to_indices
        
    def apply_hmask(self, img):
        raise FileNotFoundError('Heuristically generated masks are not found')
        # Apply random mask
        # img.save('orig.png')
        arr = np.array(img.getdata())
        max_val = np.max(arr) - 4
        
        mask_file = os.path.join(self.hmask_dir, random.choice(self.hmask_files))
        mask = Image.open(mask_file).convert('L')
        mask.resize(img.size)
        arr_mask = np.array(mask.getdata())
        noisy_mask = (arr_mask / 255) * max_val + np.random.normal(0, 8, arr_mask.shape[0])
        
        mask_idx = arr_mask > 64
        arr[mask_idx] = noisy_mask[mask_idx]
        img.putdata(arr)
        # img.save(f'masked.png')
        # exit()
        return img
    
    def apply_smask(self, img):
        img = TF.to_tensor(img)
        img = self.transform_random_erasing(img)
        img = TF.to_pil_image(img)
        return img
        
    def get_replace_path(self, path):
        # Replace the B image with another of the same glyph
        glyph = self.get_glyph(path)
        indices = self.glyph_to_indices[glyph]
        _, replace_path = self.AB_paths[random.choice(indices)]
        # while replace_path == path:
        #     _, replace_path = self.AB_paths[random.choice(indices)]
        return replace_path

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # AB_path = self.AB_paths[index % len(self.AB_paths)]
        # AB = Image.open(AB_path).convert('L')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))
        path_a, path_b = self.AB_paths[index % len(self.AB_paths)]
        A = Image.open(path_a).convert('L')
        
        if self.phase == 'train':
            # Data augmentation with Heuristically generated mask method
            if random.random() < self.hmask_prob:
                A = self.apply_hmask(A)
            
            # Data augmentation with Replace method
            if random.random() < self.replace_prob:
                path_b = self.get_replace_path(path_b)

            # Data augmentation with simple mask
            A = self.apply_smask(A)
        B = Image.open(path_b).convert('L')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': path_a, 'B_paths': path_b}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
