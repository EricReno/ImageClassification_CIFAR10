import os
import cv2
import time
import torch
import numpy
import torch.nn as nn
from config import parse_args
from model.resnet import Resnet
import torch.nn.functional as F
from dataset.cifar import CIFAR10
from dataset.augment import ToTensor

def prediction():
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
    ckpt_path = os.path.join(args.root, args.project, 'results', 'best.pth')
    ckpt_state_dict = torch.load(ckpt_path, map_location='cpu')["model"]
    model.load_state_dict(ckpt_state_dict)
    model = model.to(device)
    
    # --------------------- Build Prediction --------------------
    model.eval()
    with torch.no_grad():
        for i in range(len(val_dataset)):
            for j in range(20):
                start_time = time.time()
                image, label = val_dataset.__getitem__(i)
                image = image.unsqueeze(dim = 0).to(device)
                outputs = model(image)
                predictions = F.softmax(outputs, dim=1)
                prediction_class = torch.argmax(predictions[0]).cpu().numpy()

                show_image = val_dataset.getimage(i)
                show_image = cv2.resize(show_image, (320, 320))

                text1 = "labe:%s"%(args.class_names[int(label.numpy()[0])])
                (text_width, text_height), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                cv2.rectangle(show_image, (10, 10),  (10 + text_width, 10 + text_height), (0, 64, 255), -1) 
                cv2.putText(show_image, text1, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
                text2 = "pred:%s"%(args.class_names[prediction_class])
                (text_width, text_height), baseline = cv2.getTextSize(text2, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                cv2.rectangle(show_image, (10, 40),  (10 + text_width, 40 + text_height), (0, 128, 255), -1) 
                cv2.putText(show_image, text2, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

                time_elapsed = time.time() - start_time
                fps = 1 / time_elapsed  
                cv2.putText(show_image, 'FPS: '+str(round(fps,2)), (10, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

                cv2.imshow('image', show_image)
              
                # 退出循环的按键（通常是'q'键）  
                if cv2.waitKey(1) == ord('q'):  
                    break

        cv2.destroyAllWindows()
            
if __name__ == "__main__":
    prediction()