from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from dataloader import TorchDataset, Dataset, DataLoader
from utils.Metrics import *
from utils.tools import *
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
# import CONTA.pseudo_mask.net.resnet50_cam as resnet50_cam
from model import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


train_filename = "/home/yuxiao/public/my_train_label.txt"
val_filename = "/home/yuxiao/public/my_val_label.txt"
test1_filename = "/home/yuxiao/public/fake_test1_label.txt"

image_dir = '/home/yuxiao/public/img_dir/train'
test1_image_dir = '/home/yuxiao/public/img_dir/test1'

pred_test1_file = ""
model_name = "vgg"

root_dir = '/home/yuxiao/CS701/logs'

print_freq = 50
num_classes = 103
batch_size = 32
num_epochs = 50
pre_trained = True
feature_extract = False

threshold = 0.5

learning_rate = 0.00001


def test_model(model, dataloader, label_file):
    # label from 1-103
    model.eval()  # Set model to evaluate mode
    iter = 0
    f1 = []
    f1_epoch = []
    file = open(label_file, 'w')
    for inputs, _, image_name in dataloader:
        iter += 1
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.round(torch.sigmoid(outputs))
        preds_, index = torch.max(torch.sigmoid(outputs), 1)
        # preds = make_pred_list(num_classes, torch.sigmoid(outputs), threshold=threshold)
        for i in range(preds.shape[0]):
            file.write(image_name[i])
            labels = torch.nonzero(preds[i])
            # if labels.shape[0] == 0:
            #     file.write(" " + str(index[i].item() + 1))
            for label in labels:
                file.write(" " + str(label.item() + 1))
            file.write("\n")
        a = 0
    file.close()


def val_model(model, dataloader, criterion, optimizer, logger):
    temp1 = []
    temp2 = []
    temp3 = []
    for phase in ['val']:
        model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        iter = 0
        f1 = []
        precison = []
        recall = []
        f1_epoch = []
        precision_epoch = []
        recall_epoch = []
        for inputs, labels, _ in dataloader:
            iter += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # statistics

            running_loss += loss.item() * inputs.size(0)
            # temp1.append(outputs.detach().cpu().numpy())
            preds = make_pred_list(num_classes, torch.sigmoid(outputs), threshold=threshold)
            f, p, r = f1_score(preds.cpu().detach().numpy(), labels.data.cpu().detach().numpy())
            # temp2.append(preds.detach().cpu().numpy())
            # temp3.append(labels.detach().cpu().numpy())
            # f1.extend(f1_score(preds.cpu().detach().numpy(), labels.data.cpu().detach().numpy()))
            f1_epoch.extend(f)
            precision_epoch.extend(p)
            recall_epoch.extend(r)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_f1 = sum(f1_epoch) / len(f1_epoch)
        epoch_precision = sum(precision_epoch) / len(precision_epoch)
        epoch_recall = sum(recall_epoch) / len(recall_epoch)

        logger.info('{} Loss: {:.8f}'.format(phase, epoch_loss))
    # np.save('temp1.npy', np.concatenate(temp1, axis=0))
    # np.save('temp2.npy', np.concatenate(temp2, axis=0))
    # np.save('temp3.npy', np.concatenate(temp3, axis=0))
    logger.info('val Acc: {:.8f}'.format(epoch_f1))
    logger.info('val precision: {:.8f}'.format(epoch_precision))
    logger.info('val recall: {:.8f}'.format(epoch_recall))
    return epoch_f1


def train_model(model, dataloaders, criterion, optimizer, scheduler, logger, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
        # for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            iter = 0
            f1 = []
            precison = []
            recall = []
            f1_epoch = []
            precision_epoch = []
            recall_epoch = []
            for inputs, labels, _ in dataloaders[phase]:
                iter += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                preds = make_pred_list(num_classes, torch.sigmoid(outputs), threshold=threshold)
                f, p, r = f1_score(preds.cpu().detach().numpy(), labels.data.cpu().detach().numpy())
                f1.extend(f)
                f1_epoch.extend(f)
                precison.extend(p)
                precision_epoch.extend(p)
                recall.extend(r)
                recall_epoch.extend(r)

                if (iter + 1) % print_freq == 0:
                    logger.info('{} f1: {:.4f}'.format(iter, sum(f1) / len(f1)))
                    # logger.info(f1)
                    # print(torch.sigmoid(outputs[0]))
                    f1 = []
                    precison = []
                    recall = []

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_f1 = sum(f1_epoch) / len(f1_epoch)
            epoch_precision = sum(precision_epoch) / len(precision_epoch)
            epoch_recall = sum(recall_epoch) / len(recall_epoch)

            logger.info('{} Loss: {:.8f}'.format(phase, epoch_loss))
            logger.info('{} f1: {:.8f}'.format(phase, epoch_f1))
            logger.info('{} precision: {:.8f}'.format(phase, epoch_precision))
            logger.info('{} recall: {:.8f}'.format(phase, epoch_recall))
            logger.info('lr: {:.8f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))


            val_f1 = val_model(model, dataloaders['val'], criterion, optimizer, logger)
            # deep copy the model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                val_acc_history.append(epoch_f1)

        print()
        scheduler.step()

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_f1))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # val_model(model, dataloaders['val'], criterion, optimizer, logger)

    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    log_dir = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if not mkdir(os.path.join(root_dir, log_dir)):
        print("Fail to make log directory")
        exit()
    import logging

    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
    fhlr = logging.FileHandler(os.path.join(root_dir, log_dir, "log.txt"))  # 输出到文件的handler
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    pred_test1_file = os.path.join(root_dir, log_dir, "label.txt")

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=pre_trained)
    # model_ft = ResNet18_GAP()
    print("Initializing Datasets and Dataloaders...")

    train_data = TorchDataset(filename=train_filename, image_dir=image_dir, transform=transforms.Compose([
            transforms.Resize(224),
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    val_data = TorchDataset(filename=val_filename, image_dir=image_dir, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    test1_data = TorchDataset(filename=test1_filename, image_dir=test1_image_dir, transform= transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    datasets = {'train': train_data,
                'val': val_data}
    dataloaders_dict = {x: DataLoader(dataset=datasets[x], batch_size=batch_size, shuffle=True, num_workers=40) for x in ['train', 'val']}
    test1_dataloader = DataLoader(dataset=test1_data, batch_size=batch_size, shuffle=False, num_workers=40)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                print("\t", name)

    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    # optimizer_ft = optim.Adam(params_to_update, lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, 0.97)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    # criterion = f1_loss()
    # criterion = mixed_loss('sum', [1000.0, 1.0])
    logger.info("model size (total): " + str(get_parameter_number(model_ft)['Total']))
    logger.info("model size (trainable): " + str(get_parameter_number(model_ft)['Trainable']))
    logger.info("batch size: %d" % batch_size)
    logger.info("model: %s" % model_name)
    logger.info("pretrained: %s" % pre_trained)
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler, logger, num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))
    torch.save(model_ft, os.path.join(root_dir, log_dir, "best.pth.tar"))
    # model_dict = torch.load('/home/yuxiao/CS701/logs/2021-10-13_18:14:52/best.pth.tar').state_dict()
    # model_ft.load_state_dict(model_dict)
    logger.info(hist)
    val_model(model_ft, dataloaders_dict['val'], criterion, optimizer_ft, logger)
    # test_model(model_ft, test1_dataloader, pred_test1_file)
