import os
import cv2
import torch
import numpy
import pandas as pd
import torch.nn as nn
import seaborn as sns
from config import parse_args
from model.resnet import Resnet
import torch.nn.functional as F
from dataset.cifar import CIFAR10
from dataset.augment import ToTensor
from matplotlib import pyplot as plt

def eval():
    args = parse_args()
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # -------------------- Build Data --------------------
    transform = ToTensor()
    val_dataset = CIFAR10(data_dir = os.path.join(args.root, args.data),
                          transform = transform,
                          is_train = False)
    
    # -------------------- Build Model --------------------
    model = Resnet(args)
    ckpt_path = os.path.join(args.root, args.project, 'results', '39.pth')
    ckpt_state_dict = torch.load(ckpt_path, map_location='cpu')["model"]
    model.load_state_dict(ckpt_state_dict)
    model = model.to(device)
    
    # --------------------- Build Prediction --------------------
    correct = 0  
    confusion_matrix = numpy.zeros((args.num_classes, args.num_classes), dtype=int)  
    model.eval()
    with torch.no_grad():
        for i in range(len(val_dataset)):
            image, label = val_dataset.__getitem__(i)
            image = image.unsqueeze(dim = 0).to(device)
            outputs = model(image)
            predictions = F.softmax(outputs, dim=1)
            prediction_class = int(torch.argmax(predictions[0]).cpu().numpy())

            if prediction_class == int(label.numpy()[0]):
                correct += 1  
            confusion_matrix[prediction_class][int(label.numpy()[0])] += 1

        acc = round(correct / len(val_dataset) * 100, 2)
        print(f'Acc: {acc}%')

    confusion_matrix = (confusion_matrix/1000).round(2)
    df = pd.DataFrame(confusion_matrix, index=args.class_names, columns=args.class_names)

    plt.figure(figsize=(7.5, 6.3))
    ax = sns.heatmap(df, 
                     xticklabels=df.corr().columns, 
                     yticklabels=df.corr().columns, 
                     cmap='magma', linewidths=6, annot=True)
    
    plt.xticks(fontsize=16,family='Times New Roman')
    plt.yticks(fontsize=16,family='Times New Roman')
    

    plt.tight_layout()
    plt.savefig('results/result.jpg', transparent=False, dpi=800)
    plt.show()

if __name__ == "__main__":
    eval()