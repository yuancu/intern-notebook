import os
import glob
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    '''
    A dataset loads images from multiple directories. 
    Only png images are loaded.
    '''
    def __init__(self, transform, *folders):
        '''
        Parameters:
            transform (callable, optional): Optional transform to be applied
                on a sample.
            folders (tuple): A list of folders where images are loaded from
        '''
        super().__init__()
        self.transform = transform
        self.img_paths = []
        for folder in folders:
            folder_img_paths = glob.glob(os.path.join(folder, '*.png'))
            self.img_paths += folder_img_paths
            if len(folder_img_paths) == 0:
                print(f'WARNING: No png images detected in {folder}')

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = io.imread(self.img_paths[index])         
        image = image[:,:,:3] # drop alpha channel
#         image = torch.tensor(image, dtype=torch.float) 
#         image = image.permute((2, 0, 1)) # convert (w, h, c) to (c, w, h)
#         image = np.transpose(image, (2, 0, 1))
        if self.transform:
            image = self.transform(image)
        return image


from torchvision import transforms 
if __name__ == '__main__':
    '''test rawdata.py'''
    raw_dataset = ImageDataset(transforms.Resize((100, 100)), '../../../02_data/asset/art', '../../../02_data/asset/shading')
    print('dataset len:', len(raw_dataset))
    print('raw_dataset[0].shape', raw_dataset[0].shape)