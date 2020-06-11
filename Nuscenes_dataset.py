import os
import pandas as pd
import numpy as np

import torch
import torchvision
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class NuscenesDataset(Dataset):
    def __init__(self, NuscenesBaseDir, height=256, width=256, train=True):
        # path to main dataset
        self.baseDir = NuscenesBaseDir

        self.height, self.width = height, width

        # train = True if train dataset, else train = False
        self.train = train
        
        # Length of dataset
        if self.train:
            self.len = 26*10*12
        else:
            self.len = 26*2*12

    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        # Get the kitti sequence no
        if self.train:
            sceneNum = np.floor(idx/(26*12))
            seqNum = np.floor(idx / 12) - sceneNum*26
        else:
            sceneNum = np.floor(idx/(26*12)) + 10
            seqNum = np.floor(idx / 12) - (sceneNum-10)*26
        
        numFrames = 12
        # Get the Current frame
        frame_num = idx % 12
        
        file_path = os.path.join(self.baseDir, "scene_" + str(int(sceneNum)), "sequence_" + str(int(seqNum)), "frame_" + str(frame_num) + ".npy")
        inpTensor = np.load(file_path)
        inpTensor = torch.from_numpy(inpTensor)

        endOfSequence = False
        if frame_num == 11:
            endOfSequence = True
    
        return inpTensor, sceneNum, seqNum, frame_num, endOfSequence