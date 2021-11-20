import numpy as np
import glob

def read_txt(file_path):
    all_imgs = []
    with open(file_path, 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data

        for line in data:
            odom = line.strip().split(' ')  # 将单个数据分隔开存好

            file_name = odom[0][:-4]
            all_imgs.append(file_name)
    return all_imgs

def write_txt(file_path, files):
    my_open = open(file_path, 'a')
    for img in files:
        my_open.write(img+'\n')
    my_open.close()

def list_files_in_dir(path):
    val_imgs = []
    all_imgs = glob.glob(path+"*.jpg")
    for img in all_imgs:
        token = img.split('/')
        val_imgs.append(token[-1][:-4])
    return val_imgs

def get_all_labels(file_path):
    CAT_LIST = []
    with open(file_path, 'r') as f:
        data = f.readlines()
        for line in data:
            odom = line.strip().replace('\n','').split(' ')
            cat = ' '.join(odom[1:])
            CAT_LIST.append(cat)
    print(CAT_LIST)

def create_cls_labels_npy(file_path):

    def label2vec(labels):
        label_vec = np.zeros(103)
        for i in labels:
            label_vec[i-1]=1.0
        return label_vec


    cls_labels_dic = {}
    with open(file_path, 'r') as f:
        data = f.readlines()

        for line in data:
            odom = line.strip().replace('\n','').split(' ')
            labels = []
            k = None
            for i, v in enumerate(odom):
                if i==0:
                    k = v[:-4]
                else:
                    labels.append(int(v))
            label_npy = label2vec(labels)
            cls_labels_dic[k]=label_npy
    np.save('../food103_cls_labels.npy', cls_labels_dic)
    return cls_labels_dic


if __name__ == '__main__':
    # train_label = read_txt('./train_label.txt')
    # write_txt('food103_train.txt',train_label)

    # val_label = list_files_in_dir('/Users/daniel/Desktop/CS701-DLVision/public/img_dir/test1/')
    # write_txt('../food103_val.txt', val_label)

    # get_all_labels('../../FOOD103/classes.txt')

    cls_labels_dic = create_cls_labels_npy('./train_label.txt')
    print(cls_labels_dic)
