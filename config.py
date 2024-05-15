import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification CIFAR10')

    """
    General configuration
    """
    parser.add_argument('--cuda', default=True, help='Weather use cuda.')
    parser.add_argument('--root', default='E:\\', help='The root directory where code and data are stored')
    parser.add_argument('--data', default='data/CIFAR10', help='The path where the dataset is stored')
    parser.add_argument('--project', default='ImageClassification_CIFAR10', help='The path where the project code is stored')
    parser.add_argument('--num_classes', default=10, help='The number of the classes')
    parser.add_argument('--class_names', default= ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
                                         help= 'The category of predictions that the model can cover')
    parser.add_argument('--backbone', default='resnet18', help='The backbone network that will be used')
    

    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate.')
    parser.add_argument('--batch_size', default=256, help='The batch size used by a single GPU during training')
    parser.add_argument('--max_epoch', default=60, help='The maximum epoch used in this training')
    parser.add_argument('--pretrained', default=True, help='Whether to use pre-training weights')
    parser.add_argument('--weight', default='event2/90.pth', type=str, help="Trained state_dict file path")

    return parser.parse_args()