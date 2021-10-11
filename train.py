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

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


train_filename = "/home/yuxiao/public/train_label.txt"
test1_filename = "/home/yuxiao/public/fake_test1_label.txt"
image_dir = '/home/yuxiao/public/img_dir/train'
test1_image_dir = '/home/yuxiao/public/img_dir/test1'
pred_test1_file = ""
model_name = "resnet"
root_dir = '/home/yuxiao/CS701/logs'
print_freq = 50
num_classes = 103
batch_size = 32
num_epochs = 200
pre_trained = True
feature_extract = True

threshold = 0.5

learning_rate = 0.01


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
        preds = make_pred_list(num_classes, torch.sigmoid(outputs), threshold=threshold)
        for i in range(preds.shape[0]):
            file.write(image_name[i])
            labels = torch.nonzero(preds[i])
            for label in labels:
                file.write(" " + str(label.item() + 1))
            file.write("\n")
        a = 0
    file.close()


def val_model(model, dataloader, criterion, optimizer, logger):
    temp = []
    for epoch in range(1):
        for phase in ['val']:
            model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            iter = 0
            f1 = []
            f1_epoch = []
            for inputs, labels, _ in dataloader:
                iter += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # statistics

                running_loss += loss.item() * inputs.size(0)
                temp.append(outputs.detach().cpu().numpy())
                preds = make_pred_list(num_classes, torch.sigmoid(outputs), threshold=threshold)
                f1.extend(f1_loss(preds.cpu().detach().numpy(), labels.data.cpu().detach().numpy()))
                f1_epoch.extend(f1_loss(preds.cpu().detach().numpy(), labels.data.cpu().detach().numpy()))

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_f1 = sum(f1_epoch) / len(f1_epoch)

            logger.info('{} Loss: {:.8f}'.format(phase, epoch_loss))
    np.save( 'temp.npy', np.concatenate(temp, axis=0))
    logger.info('Best val Acc: {:4f}'.format(epoch_f1))


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
            f1_epoch = []
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
                f1.extend(f1_loss(preds.cpu().detach().numpy(), labels.data.cpu().detach().numpy()))
                f1_epoch.extend(f1_loss(preds.cpu().detach().numpy(), labels.data.cpu().detach().numpy()))

                if (iter + 1) % print_freq == 0:
                    logger.info('{} f1: {:.4f}'.format(iter, sum(f1) / len(f1)))
                    # logger.info(f1)
                    # print(torch.sigmoid(outputs[0]))
                    f1 = []

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_f1 = sum(f1_epoch) / len(f1_epoch)

            logger.info('{} Loss: {:.8f}'.format(phase, epoch_loss))
            logger.info('{} f1: {:.8f}'.format(phase, epoch_f1))
            logger.info('lr: {:.8f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

            # deep copy the model
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                val_acc_history.append(epoch_f1)

        print()
        scheduler.step()

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # val_model(model, dataloaders['val'], criterion, optimizer, logger)

    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


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
    print("Initializing Datasets and Dataloaders...")

    train_data = TorchDataset(filename=train_filename, image_dir=image_dir, transform=transforms.Compose([
            transforms.Resize(224),
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    test1_data = TorchDataset(filename=test1_filename, image_dir=test1_image_dir, transform= transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    dataloaders_dict = {x: DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=40) for x in ['train', 'val']}
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
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, 0.97)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    logger.info("model size (total): " + str(get_parameter_number(model_ft)['Total']))
    logger.info("model size (trainable): " + str(get_parameter_number(model_ft)['Trainable']))
    logger.info("batch size: %d" % batch_size)
    logger.info("model: %s" % model_name)
    logger.info("pretrained: %s" % pre_trained)
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler, logger, num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))
    torch.save(model_ft, os.path.join(root_dir, log_dir, "best.pth.tar"))
    logger.info(hist)
    val_model(model_ft, dataloaders_dict['val'], criterion, optimizer_ft, logger)
    test_model(model_ft, test1_dataloader, pred_test1_file)
