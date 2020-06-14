import os
import pandas as pd
import numpy as np

import torch
import torchvision
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class NuscenesDatasetTest(Dataset):
    def __init__(self, NuscenesBaseDir, height=256, width=256):
        # path to main dataset
        self.baseDir = NuscenesBaseDir

        self.height, self.width = height, width
        
        # Length of dataset
        self.len = 26*9*12

    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        sceneNum = np.floor(idx/(26*12)) + 44
        seqNum = np.floor(idx / 12) - (sceneNum-44)*26
        
        numFrames = 12
        # Get the Current frame
        frame_num = idx % 12
        
        file_path = os.path.join(self.baseDir, "scene_" + str(int(sceneNum)), "sequence_" + str(int(seqNum)), "frame_" + str(frame_num) + ".npy")
        inpTensor = np.load(file_path)
        
        for i in range(len(inpTensor)):
            #import pdb; pdb.set_trace()
            inpTensor[i,:,:] = ((inpTensor[i,:,:] / np.max(inpTensor[i,:,:]))*255).astype(int)
            
        inpTensor = torch.from_numpy(inpTensor)

        endOfSequence = False
        if frame_num == 11:
            endOfSequence = True
    
        return inpTensor, sceneNum, seqNum, frame_num, endOfSequence