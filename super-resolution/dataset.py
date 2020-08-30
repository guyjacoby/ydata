import os
import cv2
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    
    def __init__(self, filelist, transforms=None):
        super().__init__()
        self.filelist = filelist
        self.transforms = transforms
        
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        
        path, filename = os.path.split(self.filelist[idx])
        X_img = cv2.imread(path + '/X/' + filename)
        y_mid_img = cv2.imread(path + '/y_mid/' + filename)
        y_large_img = cv2.imread(path + '/y_large/' + filename)
        
        if self.transforms is not None:
            X_img = self.transforms(X_img)
            y_mid_img = self.transforms(y_mid_img)
            y_large_img = self.transforms(y_large_img)
            
        return X_img, y_mid_img, y_large_img