import os
import time
import torch
import torch.nn as nn

from config import parse_args
import torch.nn.functional as F
from model.resnet import Resnet
from dataset.cifar import CIFAR10
from dataset.augment import ToTensor
from torch.optim.lr_scheduler import MultiStepLR  
from torch.utils.tensorboard import SummaryWriter 

def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("--------------------------------------------------------")

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # -------------------- Build Data --------------------
    transform = ToTensor()

    val_dataset = CIFAR10(data_dir = os.path.join(args.root, args.data),
                          transform = transform,
                          is_train = False)
    train_dataset = CIFAR10(data_dir = os.path.join(args.root, args.data),
                            transform = transform,
                            is_train = True)
 
    val_sampler = torch.utils.data.RandomSampler(val_dataset)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, args.batch_size, drop_last=True)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_batch_sampler, pin_memory=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, pin_memory=True)

    # -------------------- Build Model --------------------
    model = Resnet(args)
    model = model.to(device)
    
    # -------------------- Build Loss --------------------
    criterion = nn.CrossEntropyLoss()  

    # -------------------- Build Optimizer --------------------
    learning_rate = (args.batch_size/64)*args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.937, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    writer = SummaryWriter('results/log')

    # --------------------- Build Train --------------------
    start = time.time()
    max_acc = 0
    for epoch in range(args.max_epoch):
        ckpt_path = os.path.join(args.root, args.project, 'results', '{}.pth'.format(epoch))

        model.train()
        train_loss = 0.0 
        for iteration, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.squeeze(dim=1).to(device)

            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(images)

            loss = criterion(outputs, labels.long())  # 计算损失
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            train_loss += loss

            print("Epoch [{}:{}/{}:{}], Time [{}] Loss: {:.4f}".format(epoch, args.max_epoch, iteration, len(train_dataloader), 
                                                                       time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), loss))
        
        # scheduler.step()
        writer.add_scalar('Loss/Train', train_loss/len(train_dataloader), epoch)

        model.eval()
        correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for iteration, (images, labels) in enumerate(val_dataloader):
                images = images.to(device)
                labels = labels.squeeze(dim=1).to(device)

                outputs = model(images)  
                loss = criterion(outputs, labels.long())  # 计算损失
                val_loss += loss

                predictions = F.softmax(outputs, dim=1)
                _, predicted_classes = torch.max(predictions, 1)
                correct += (predicted_classes == labels).float().sum().item() 

            writer.add_scalar('Loss/Val', val_loss/len(val_dataloader), epoch)
            acc = round(correct / len(val_dataset) * 100, 2)
            print(f'Acc: {acc}%')
            
        if acc > max_acc:
            max_acc = acc
            ckpt_path = os.path.join(args.root, args.project, 'results', '{}.pth'.format(epoch+1))
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args},
                        ckpt_path)

if __name__ == "__main__":
    train()