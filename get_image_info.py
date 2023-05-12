import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
import pandas as pd
import math
import glob
def main():
    args = parse.parse_args()
    test_list = args.test_list
    batch_size = args.batch_size
    model_path = args.model_path
    torch.backends.cudnn.benchmark=True
    test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'],get_feature=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    test_dataset_size = len(test_dataset)

    print('the number of test image is ', test_dataset_size)
    corrects = 0
    acc = 0
    #model = torchvision.models.densenet121(num_classes=2)
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.cuda()
    model.eval()


    file_image_dir = "/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/incremental_learning_FF++_dataset/NT_real2500_fake2500_train.txt"
    res = [] 

    image_label = []
    image_confidence = []
    image_margin = []
    image_entropy = []
    image_pred = []
    
    fh = open(file_image_dir, 'r')
    for line in fh:
        line = line.rstrip()
        words = line.split(',')

        image_file = words[1]
        path = os.path.join(image_file,'*.png')
        image_filenames = sorted(glob.glob(path))

        for i in image_filenames:
            res.append(i)
            image_label.append(int(words[0]))
    print(len(image_label))
    print(len(res))
    # exit()
    
    sum = 0

    bug = 0
    with torch.no_grad():
        for (image, labels) in test_loader:
            image = image.cuda()
            labels = labels.cuda()
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            prob = nn.functional.softmax(outputs.data,dim=1)

            # calculate the difficult of sample 

            
            for i in range(len(image)):
                bug += 1
                print(bug)
                if outputs[i][0].item() <= -40:
                    prob[i][0] = 1e-40
                if outputs[i][0].item() <= -40:
                    print("yes!!")
                    prob[i][1] = 1e-40

                image_confidence.append(max(prob[i][0].item(),prob[i][1].item()))
                image_margin.append(max(prob[i][0].item(),prob[i][1].item()) - min(prob[i][0].item(),prob[i][1].item()))
                print(outputs[i][0].item(),outputs[i][1].item())
                print(prob[i][0].item(),prob[i][1].item())
                image_entropy.append(- (prob[i][0].item() * math.log(prob[i][0].item()) + prob[i][1].item() * math.log(prob[i][1].item())) )
                if(prob[i][0]>prob[i][1]):
                    image_pred.append(0)
                else:
                    image_pred.append(1)


            corrects += torch.sum(preds == labels.data).to(torch.float32)
            print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
        acc = corrects / test_dataset_size
        print('Test Acc: {:.4f}'.format(acc))
    print(len(test_loader))
    print(len(res))
    print(len(image_confidence))
    print(len(image_margin))
    print(len(image_entropy))
    print(len(image_pred))
    print(len(image_label))
    dict = {'image_info': res, 'image_confidence': image_confidence, 'image_margin': image_margin, 'image_entropy':image_entropy, 'image_pred':image_pred,'image_label':image_label}
    df = pd.DataFrame(dict)
 
    #保存 dataframe
    df.to_csv('incremental_learning_task3_NT(2500real_2500fake)_20230215_info.csv')



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=128)
    #parse.add_argument('--test_list', '-tl', type=str, default='./data_list/Deepfakes_c0_test.txt')
    parse.add_argument('--test_list', '-tl', type=str, default='/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/incremental_learning_FF++_dataset/NT_real2500_fake2500_train.txt')
    #parse.add_argument('--model_path', '-mp', type=str, default='./pretrained_model/df_c0_best.pkl')
    parse.add_argument('--model_path', '-mp', type=str, default='/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/output/using_NT(2500real_and_2500fake)_train_model/best.pkl')
    
    main()

    print('Hello world!!!')