import torch
import numpy
class ToTensor():
    def __call__(self, 
                 image, 
                 label
                 ):
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        label_tensor = torch.tensor([label], dtype=torch.float) 
        
        return image_tensor, label_tensor
