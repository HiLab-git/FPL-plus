import numpy as np
import csv
# all_names_dict = np.load('dataset/weight/vs_t2s.npy',allow_pickle=True)
all_names_dict = np.load('dataset/weight/cyc121_vst1s-gan.npy',allow_pickle=True)
print(len(all_names_dict))
print(all_names_dict)
eva_filenames = []
tra_filenames = []
all_weights = []
for i in range(int(1*len(all_names_dict))):
    image_weight = all_names_dict[i][0][0]
    if image_weight !=1:
        all_weights.append(image_weight)
max = max(all_weights)
min = min(all_weights)
print('max weight value:',max,'; min weight value:',min)
for i in range(int(1*len(all_names_dict))):
    sig_dir = all_names_dict[i][1]
    img_name_eva = sig_dir.split('/')[-1]#.replace('.nii.gz','_seg.nii.gz')
    lab_name_eva = sig_dir.split('/')[-1]
    img_name = sig_dir
    lab_name = sig_dir.replace('./dataset/hrT2_train/img','./results_dual/vs_t1s_g_i-train_hrT2')
    weight_pixel  = sig_dir.replace('./dataset/hrT2_train/img','dataset/hrT2_pixel-weight')
    image_weight = all_names_dict[i][0][0]
    if image_weight>max:
        image_weight = max
    image_weight = abs((max - image_weight)/(max-min))+0.01
    print('image weight:',image_weight)
    tra_filenames.append([img_name, lab_name, weight_pixel, image_weight])
tra_output_file = 'config_dual/data_vs/train_vs_t1s_wi+wp.csv'
fields      = ['image', 'label' , 'pixel_weight','image_weight']
with open(tra_output_file, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', 
                        quotechar='"',quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(fields)
    for item in tra_filenames:
        csv_writer.writerow(item)