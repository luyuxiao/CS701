import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from misc import torchutils, imutils

cudnn.enabled = True

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        # model.cpu()
        model.cuda()
        model.eval()
        file_names = os.listdir('/data/yuxiao/cam_ass_aug/result/cam')
        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            # if size[0] > 1500 or size[1] > 1500:
            #     continue

            if img_name + '.npy' in file_names:
                continue
            print(img_name)
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            with torch.no_grad():
                outputs = [model(img[0].cuda(non_blocking=True)).cpu()
                           for img in pack['img']]
            # outputs = [model(img[0].cpu())
            #            for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]

            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            # strided_cam = strided_cam[valid_cat].cpu().numpy()
            # temp = strided_cam.max(axis=0)
            # strided_cam /= temp + 1e-5
            #
            # highres_cam = highres_cam[valid_cat].cpu().numpy()
            # temp = highres_cam.max(axis=0)
            # highres_cam /= temp + 1e-5
            #
            # # save cams
            # np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
            #         {"keys": valid_cat, "cam": strided_cam, "high_res": highres_cam})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')



def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM_101_ENSEMBLE')()
    # model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    print(args.cam_weights_name + '.pth')
    model.load_state_dict(torch.load('/data/yuxiao/cam_ass_aug/sess/res101_cam_100.pth'), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root,
                                                             sub_dir='train',
                                                             scales=args.cam_scales)

    # dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.val_list,
    #                                                          voc12_root=args.voc12_root,
    #                                                          sub_dir='val',
    #                                                          scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()