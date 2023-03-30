
import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from PIL import Image
from itertools import combinations
import pickle
import numpy as np



class OriginalReconstructionDataset(Dataset):
    DATASET_DIR='./BIRN_dataset/'
    all_radial_lines=[20,40,60,80,100]
    images_dir=(DATASET_DIR+'birn_png/')
    rectype=['L2','L1','TV']
    rectypes_list=[]
    for i in range(len(rectype)):
        for p in combinations(rectype, i+1):  # 2 for pairs, 3 for triplets, etc
            rectypes_list.append('_'.join(p))
    

    def __init__(self, radial_lines, rectype, set='train', img_size=(256,256), return_idx=False):
        self.get_indices()
        self.rectype=rectype.split('_')
        self.radial_lines=radial_lines
        self.images_dir=(self.DATASET_DIR+'birn_png/')
        self.set=set
        self.return_idx=return_idx
        rec_dirs=[(f"{self.DATASET_DIR}birn_pngs_{rl}lines_{rt}/") for rt in self.rectypes_list for rl in self.all_radial_lines]
        
        self.rec_images_dirs=[]
        for dir in rec_dirs:
            for rt in self.rectype:
                if rt in dir:
                    if str(self.radial_lines) in dir:
                        self.rec_images_dirs.append(dir)
                        break

        self.images = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
        if set=='train':
            self.images = [self.images[i] for i in self.train_indexes]
        if set=='val' or set=='validation':
            self.images = [self.images[i] for i in self.val_indexes]
        if set=='test':
            self.images = [self.images[i] for i in self.test_indexes] 
        self.transform = transforms.Compose([
                        transforms.Grayscale(num_output_channels=1),         
                        transforms.Resize(img_size),
                        #transforms.Lambda(lambda x: x/255.0),
                        transforms.ToTensor()
                        ])
        
        
    def __len__(self):
    # return length of image samples    
        return len(self.images)

    def __getitem__(self, idx):
        img_name=self.images[idx]
        img = Image.open(self.images_dir+img_name)
        img=self.transform(img)#.half()
        rec_imgs=[]
        for rec,dir in zip(self.rectype,self.rec_images_dirs):
            noisy_name=img_name[:-14]+rec+f'_{self.radial_lines}lines.png'            
            tensor=self.transform(Image.open(dir+noisy_name))
            rec_imgs.append(tensor)
        noisy=torch.stack(rec_imgs)
        noisy=torch.squeeze(noisy, 1)#.half()
        if self.return_idx:
            return (img,noisy, idx)
        else:
            return (img,noisy)

    def print_first_element_shape(self):
        for data in self:
            print(data[0].shape)
            print(data[1].shape)
            break

    def get_indices(self):
        idx_file='indices.pkl'
        if not os.path.exists(idx_file):
            np.random.seed(seed=42)
            all_indexes=np.random.permutation(len([f for f in os.listdir(self.images_dir) if f.endswith('.png')]))
            m = len(all_indexes)
            m_train=int(m*0.8)
            m_val = int(m*0.1)
            self.train_indexes=all_indexes[:m_train]
            self.val_indexes=all_indexes[m_train:m_train+m_val]
            self.test_indexes=all_indexes[m_train+m_val:]
            
            with open(idx_file, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([self.train_indexes, self.val_indexes, self.test_indexes], f)
        else:
            with open(idx_file,'rb') as f:  # Python 3: open(..., 'rb')
                self.train_indexes, self.val_indexes, self.test_indexes = pickle.load(f)


