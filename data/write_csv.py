"""Script for writing cvs files
"""

import os
import csv
import pandas as pd
import random
from random import shuffle

def create_csv_file_vs(data_dir, output_file, fields):
    """
    create a csv file to store the paths of files for each patient
    """
    filenames = []
    image_names = os.listdir(data_dir)
    image_names = [item for item in image_names if "image" in item]
    image_names.sort()
    print('total number of images {0:}'.format(len(image_names)))
    for img_name_ in image_names:
        img_name = data_dir +'/'+ img_name_
        lab_name = img_name_.replace("image", "label")
        lab_name = data_dir+ '/' + lab_name
        filenames.append([img_name, lab_name])

    with open(output_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(fields)
        for item in filenames:
            csv_writer.writerow(item)
def create_csv_file_bst(data_dir, output_file, fields):
    """
    create a csv file to store the paths of files for each patient
    """
    filenames = []
    image_names = os.listdir(data_dir)
    image_names = [item for item in image_names if "BraTS20_Training" in item]
    image_names.sort()
    print('total number of images {0:}'.format(len(image_names)))
    for img_name_ in image_names:
        img_name = data_dir +'/'+ img_name_
        # lab_name = img_name.replace("_z", "_seg_z")
        lab_name = data_dir[:-3]+'lab/' + img_name_
        filenames.append([img_name, lab_name])

    with open(output_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(fields)
        for item in filenames:
            csv_writer.writerow(item)
def random_split_dataset00():
    random.seed(2022)
    input_file = 'config_bst/bst_all.csv'
    flare_names_file = 'config_bst/bst_flare.csv'
    flair_names_file = 'config_bst/bst_t2.csv'
    test_names_file  = 'config_bst/bst_test.csv'
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    shuffle(data_lines)
    N = len(data_lines)
    # n1 = int(N * 0.8)
    n1 = 164
    n2 = 164
    # n2 = int(N * 0.8)
    print('image number', N)
    print('training number', n1)
    print('validation number', n2)
    print('testing number', N - n2 - n1)
    flare_lines  = sorted(data_lines[:n1])
    flair_lines  = sorted(data_lines[n1:n1+n2])
    test_lines   = data_lines[n1+n2:]
    with open(flare_names_file, 'w') as f:
        f.writelines(lines[:1] + flare_lines)
    with open(flair_names_file, 'w') as f:
        f.writelines(lines[:1] + flair_lines)
    with open(test_names_file, 'w') as f:
       f.writelines(lines[:1] + test_lines)
def random_split_dataset_vs():
    random.seed(2021)
    input_file = '/data2/jianghao/VS/vs_seg2021/config/data_vs/t2_all.csv'
    train_names_file = 'config/data_vs/t2_train.csv'
    valid_names_file = 'config/data_vs/t2_valid.csv'
    # test_names_file  = 'config_bst/bst_test.csv'
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    shuffle(data_lines)
    N = len(data_lines)
    n1 = int(N / 8 * 7)
    # n1 = 164
    # n2 = int(N * 0.8)
    print('image number', N)
    print('training number', n1)
    print('validation number', N - n1)
    #print('testing number', N - n2)
    train_lines  = sorted(data_lines[:n1])
    valid_lines  = sorted(data_lines[n1:])
    #test_lines   = data_lines[n2:]
    with open(train_names_file, 'w') as f:
        f.writelines(lines[:1] + train_lines)
    with open(valid_names_file, 'w') as f:
        f.writelines(lines[:1] + valid_lines)
    #with open(test_names_file, 'w') as f:
    #    f.writelines(lines[:1] + test_lines)
def random_split_dataset():
    random.seed(2021)
    input_file = '/data2/jianghao/VS/vs_seg2021/config/data_bst/flair_all.csv'
    train_names_file = 'config/data_bst/flair_train.csv'
    valid_names_file = 'config/data_bst/flair_valid.csv'
    # test_names_file  = 'config_bst/bst_test.csv'
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    shuffle(data_lines)
    N = len(data_lines)
    n1 = int(N / 8 * 7)
    # n1 = 164
    # n2 = int(N * 0.8)
    print('image number', N)
    print('training number', n1)
    print('validation number', N - n1)
    #print('testing number', N - n2)
    train_lines  = sorted(data_lines[:n1])
    valid_lines  = sorted(data_lines[n1:])
    #test_lines   = data_lines[n2:]
    with open(train_names_file, 'w') as f:
        f.writelines(lines[:1] + train_lines)
    with open(valid_names_file, 'w') as f:
        f.writelines(lines[:1] + valid_lines)
    #with open(test_names_file, 'w') as f:
    #    f.writelines(lines[:1] + test_lines)

def get_evaluation_image_pairs(test_csv, gt_seg_csv):
    with open(test_csv, 'r') as f:
        input_lines = f.readlines()[1:]
        output_lines = []
        for item in input_lines:
            gt_name = item.split(',')[1].rstrip()
            seg_name = item.split(',')[0].rstrip()
            output_lines.append([gt_name, seg_name])
    with open(gt_seg_csv, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["ground_truth", "segmentation"])
        for item in output_lines:
            csv_writer.writerow(item)


if __name__ == "__main__":
    # create cvs file for promise 2012
    fields      = ['image', 'label']
    data_dir    = '/your/nii/data/dir'
    output_file = '/your/csv/save/dir/XX.csv'
    create_csv_file_vs(data_dir, output_file, fields)


