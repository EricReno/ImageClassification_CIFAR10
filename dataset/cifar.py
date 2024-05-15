import os
import cv2
import torch.utils.data as data

class CIFAR10(data.Dataset):
    def __init__(self,
                 data_dir : str = None,
                 transform = None,
                 is_train : bool = None) -> None:
        super().__init__()

        self.datapath = data_dir
        self.transform = transform
        self.is_train = is_train

        self.data = list()
        if self.is_train:
            for line in open(os.path.join(self.datapath, 'train.txt')):
                self.data.append(
                    {
                        'filename' : os.path.join(self.datapath, 'train', line.strip().split(' ')[0]+'.png'),
                        'label' : int(line.strip().split(' ')[1])
                    })
        else:
            for line in open(os.path.join(self.datapath, 'test.txt')):
                self.data.append(
                    {
                        'filename' : os.path.join(self.datapath, 'test', line.strip().split(' ')[0]+'.png'),
                        'label' : int(line.strip().split(' ')[1])
                    })
                
        self.dataset_size = len(self.data)
        
    def __getitem__(self, index):
        image = cv2.imread(self.data[index]['filename'], cv2.IMREAD_COLOR)
        label = self.data[index]['label']

        if not self.transform == None:
            image, label = self.transform(image, label)

        return image, label

    def getimage(self, index):
        image = cv2.imread(self.data[index]['filename'], cv2.IMREAD_COLOR)

        return image
    
    def __len__(self):
        return self.dataset_size