import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib
import numpy as np
import voc12.dataloader
from misc import pyutils, torchutils

from sklearn.metrics import precision_score,recall_score, f1_score

def preds2npy(k, v):

    def label2vec(labels):
        label_vec = np.zeros(103)
        for i in labels:
            label_vec[i - 1] = 1.0
        return label_vec

    cls_labels_dic = {}
    for idx in range(len(k)):
        jpgname = '{}'.format(k[idx])
        preds = v[idx]

        label_npy = label2vec(preds)
        cls_labels_dic[jpgname] = label_npy

    np.save('voc12/food103_cls_labels_val.npy', cls_labels_dic)

def preds2txt(k,v):
    all_ = []
    for idx in range(len(k)):
        jpgname = k[idx]
        preds = v[idx]
        one_line = jpgname #'{}.jpg'.format(jpgname)
        if preds.size==0:
            pass
        else:
            for one_cls in preds[0]:
                one_line = one_line+' {}'.format(one_cls)
        all_.append(one_line)

    my_open = open('./sess/stage1.txt', 'a')
    for img in all_:
        my_open.write(img + '\n')
    my_open.close()

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    model.eval()
    preds_list = []
    names_list = []
    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']
            x = model(img)
            pred = F.sigmoid(x)
            preds_list.extend(pred.detach().cpu().numpy())
            names_list.extend(pack['name'])

    preds_list = np.array(preds_list)
    preds = np.array(preds_list > 0.5, dtype=int)
    test_results = []
    for idx, pred in enumerate(preds):
        # pred_arg = np.squeeze(np.argwhere(pred > 0) + 1)
        pred_arg = np.argwhere(pred > 0) + 1

        if pred_arg.size == 0:
            particular_pred = preds_list[idx]
            pred_arg_temp = []
            pred_arg_temp.append(np.argmax(particular_pred))
            pred_arg = np.reshape(pred_arg_temp, (1, -1))

        else:
            pred_arg = pred_arg.reshape(1,-1)
        test_results.append(pred_arg)
    preds2txt(names_list, test_results)
    preds2npy(names_list, test_results)
    model.train()
    return

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred)
    target = np.array(target)

    pred = np.array(pred > threshold, dtype=float)
    p = {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            # 'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            # 'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            # 'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            # 'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            # 'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            # 'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }
    for k,v in p.items():
        print(str(k) + '\t' + str(v))
    # print(p['micro/f1'])
    return p['micro/f1']


def run_NAS(model, data_loader, threshold):
    preds_list = []
    gt_list = []
    model.eval()
    for step, pack in enumerate(data_loader):
        img = pack['img'].cuda(non_blocking=True)

        label = pack['label'].cuda(non_blocking=True)

        x = model(img)
        pred = torch.sigmoid(x)

        preds_list.extend(pred.detach().cpu().numpy())
        gt_list.extend(label.cpu().numpy())

    f1 = calculate_metrics(preds_list, gt_list, threshold)
    # with open('/home/yuxiao/Spearmint/examples/CS701/result.txt', 'a') as f:
    #     f.write(str(f1) + '\n')
    return f1


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net_50')()


    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                food_sub_dir='train',
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=20, drop_last=False)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,food_sub_dir='train',
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=20, drop_last=False)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model = model.cuda()
    model.train()

    # f1 = run_NAS(model, val_data_loader, args.threshold)
    # print(f1)
    # return f1
    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()
    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))
        preds_list = []
        gt_list = []

        for step, pack in enumerate(train_data_loader):
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            pred = torch.sigmoid(x)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss1': loss.item()})

            preds_list.extend(pred.detach().cpu().numpy())
            gt_list.extend(label.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

                calculate_metrics(preds_list, gt_list)
        torch.save(model.module.state_dict(), args.cam_weights_name + str(ep) + '.pth')
        print(len(preds_list), len(gt_list))

        # validate(model, val_data_loader)
        # torch.save(model.module.state_dict(), args.cam_weights_name  + '.pth')
    # torch.cuda.empty_cache()

