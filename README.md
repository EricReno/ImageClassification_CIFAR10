# Hello, Resnet18--Cifar10

## 数据集:Cifar10

- **train**: 5*10000   
- **val**: 10*1000  
- **官方网址**：https://www.cs.toronto.edu/~kriz/cifar.html
- **备注** 可通过torchvision直接下载。本实验数据转为png和txt格式（详见dataset/utils.py）

## 通用设置
| backbone | img_size | epoch | pretrained | data_augment |  
| :---:    | :---:    | :---: | :---:      | :---:        |
| resnet18 | 32 x 32  | 60    | CoCo       | False        |


## LR-收敛速度：消融实验
| Name  |learning_rate| lr_decay   |  optimizer   |  Val_loss |  Acc  |         |       |
| :---: |  :---:      | :---:      | :---:        |  :---:    | :---: |  :---:  | :---: |
|event_0| 0.01        | None       | SGD(None)    | 2.399     | 11.36%|  震荡   |        |
|event_1| 0.001       | None       | Adam(None)   | 0.994     | 71.71%| epoch38 |后续跟进|
|event_2| 0.001       | None       | SGD(None)    | 0.710     | 76.77%| epoch20 |       |
|event_3| 0.001       | None       |SGD(937, 0005)| 0.700     | 79.22%| epoch03 | √     |
|event_4| 0.0001      | None       |SGD(937, 0005)| 0.600     | 75.90%| epoch11 |       |
|event_5| 0.001       |Step(10,0.5)|SGD(937, 0005)| 0.619     | 79.24%| epoch03 | √     |


## BS-精度影响：消融实验（benchmark：event3）
| Name  |batch_size| Val_loss |  Acc  |  Epoch  | .pth  | onnx |
| :---: |  :---:   |  :---:   | :---: |  :---:  | :---: | :---:|
|event_3| 64       | 0.700    | 79.22%| epoch03 |       |      |
|event_6| 32       | 0.576    | 83.18%| epoch39 | 260M  | 130M |
|event_7| 128      | 0.679    | 82.47%| epoch50 |       |      |
|event_8| 256      | 0.960    | 82.30%| epoch60 |       |      |

<img src="results\best.jpg">
