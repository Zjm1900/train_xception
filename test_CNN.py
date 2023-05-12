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
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import f1_score

def main():
    args = parse.parse_args()
    test_list = args.test_list
    batch_size = args.batch_size
    model_path = args.model_path
    print(model_path)
    print(test_list)
    torch.backends.cudnn.benchmark=True
    test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'],get_feature = False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    test_dataset_size = len(test_dataset)
    corrects = 0
    acc = 0
    #model = torchvision.models.densenet121(num_classes=2)
    
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    #model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.cuda()
    model.eval()

    #exit()
    sum = 0
    pro_all = []
    label_all = []
    F1_prob_all = []
    with torch.no_grad():
        for (image, labels) in test_loader:

            image = image.cuda()
            labels = labels.cuda()
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            # print(outputs)
            # print(outputs[:,1].cpu().numpy())
            # exit()
            pro_all.extend(outputs[:,1].cpu().numpy())
            F1_prob_all.extend(np.argmax(outputs.cpu().numpy(),axis=1))
            label_all.extend(labels.cpu().numpy())
            corrects += torch.sum(preds == labels.data).to(torch.float32)
            #print(labels)
            print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/len(image)))
        acc = corrects / test_dataset_size
        print('Test Acc: {:.4f}'.format(acc))
    print("AUC:{:.8f}".format(roc_auc_score(label_all,pro_all)))

    print("F1-Score:{:.8f}".format(f1_score(label_all,F1_prob_all)))

if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=128)
    #parse.add_argument('--test_list', '-tl', type=str, default='./data_list/Deepfakes_c0_test.txt')

    parse.add_argument('--test_list', '-tl', type=str, default='/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/data_list/test.txt')
    #parse.add_argument('--test_list', '-tl', type=str, default='/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/DFDC_datalist/test.txt')
    # bparse.add_argument('--test_list', '-tl', type=str, default='/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/incremental_learning_FF++_dataset/DFD_dataset/Task4_DFD_test.txt')
    # parse.add_argument('--test_list', '-tl', type=str, default='/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/Celeb-DF/test.txt')
    
    # parse.add_argument('--model_path', '-mp', type=str, default='./pretrained_model/df_c0_best.pkl')
    #parse.add_argument('--model_path', '-mp', type=str, default='/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/output/using_FF++_Memory_fine-tune_model_20230216/4_demo_suffle.pkl')
    parse.add_argument('--model_path', '-mp', type=str, default='/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/output/using_CDF(2500real_2500fake)_train_model_trained_by_FF++_and_DFDC_and_DFD_with_SupCon/1_demo.pkl')
    
    main()

    print('Hello world!!!')