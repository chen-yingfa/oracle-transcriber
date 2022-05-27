from pathlib import Path
import random

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class DomainData(Dataset):
    def __init__(self, data_dir: Path, phase: str, num_examples=None, img_paths=None):
        if phase == 'train':
            self.augmentation = transforms.Compose([
                transforms.Resize(96),
                transforms.RandomCrop(84),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.augmentation = transforms.Compose([
                transforms.Resize(84),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])
        
        self.phase = phase
        if img_paths is not None:
            self.img_paths = img_paths
        else:
            self.img_paths = self.get_img_paths(data_dir)[:num_examples]
        print(f'Got {len(self.img_paths)} images')
    
    def get_img_paths(self, data_dir: Path):
        '''
        data_dir structure: 
        
        glyph0/
            img0.png
            img1.png
            ...
        glyph1/
        ...
        
        '''
        img_paths = []
        for img_path in data_dir.iterdir():
            img_paths.append(img_path)
        return img_paths
    
    def __getitem__(self, idx: int) -> dict:
        path = self.img_paths[idx % len(self.img_paths)]
        label = 1 if path.name.split('.')[0][-1] == 't' else 0
        
        img = Image.open(path).convert('RGB')
        img = self.augmentation(img)
        return {
            'img': img,
            'label': label,
        }

    def __len__(self):
        return len(self.img_paths)


class PairData(Dataset):
    def __init__(self, data_dir: Path, phase: str, num_examples=None, img_paths=None):
        if phase == 'train':
            self.augmentation = transforms.Compose([
                transforms.Resize(96),
                # transforms.RandomCrop(84),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.augmentation = transforms.Compose([
                transforms.Resize(96),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])
        
        self.phase = phase
        if img_paths is not None:
            self.a_paths, self.b_paths = img_paths
        else:
            self.a_paths, self.b_paths = self.get_img_paths(data_dir)[:num_examples]
        print(f'Got {len(self.a_paths)} images')
    
    def get_img_paths(self, data_dir: Path):
        '''
        '''
        paths = sorted(data_dir.iterdir())
        a_paths = [paths[i] for i in range(0, len(paths), 2)]   # Rubbings
        b_paths = [paths[i+1] for i in range(0, len(paths), 2)] # Transcriptions
        return a_paths, b_paths
    
    def __getitem__(self, idx: int) -> dict:
        '''
        Return rubbing, transcription. 
        
        0.5 chance to return valid pair.
        '''
        path_a = self.a_paths[idx % len(self.a_paths)]
        path_b = self.b_paths[idx % len(self.b_paths)]
        img_a = Image.open(path_a).convert('RGB')
        
        if self.phase == 'train' or self.phase == 'dev':
            if random.random() < 0.5:
                # Return real pair
                img_b = Image.open(path_b).convert('RGB')
                label = 1
            else:
                # Return fake pair
                new_path_b = random.choice(self.b_paths)
                while new_path_b == path_b:
                    new_path_b = random.choice(self.b_paths)
                img_b = Image.open(new_path_b).convert('RGB')
                label = 0
        else:
            img_b = Image.open(path_b).convert('RGB')
            label = 1

        # img_a.save(f'images/{idx}_a.png')
        # if label == 0:
        #     img_b.save(f'images/{idx}_fake_b.png')
        # else:
        #     img_b.save(f'images/{idx}_real_b.png')

        return {
            'img_a': self.augmentation(img_a),
            'img_b': self.augmentation(img_b),
            'label': label,
        }

    def __len__(self):
        return len(self.a_paths)
    
