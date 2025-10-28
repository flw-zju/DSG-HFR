from torch.utils.data import DataLoader
from tqdm import tqdm
from dual_tower import create_model
import argparse
from utils import *
import torch.nn as nn
from dataset import make_train_dataset
import os


parser = argparse.ArgumentParser()

parser.add_argument('--vis_train_text', type=str, default="./project_texts/vis_train.txt")
parser.add_argument('--inf_train_text', type=str, default="./project_texts/inf_train.txt")
parser.add_argument('--vis_root', type=str, default="./tufts_dataset/vis/train/")
parser.add_argument('--inf_root', type=str, default="./tufts_dataset/inf/train/")
parser.add_argument('--gen_vis_path', type=str, default="./generated_images/vis")
parser.add_argument('--model_type', type=str, default="DualTowerTransformer",
                    help="switch DualTowerModel or DualTowerTransformer as the recognition net")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--pretrain_epoch', type=int, default=7)
parser.add_argument('--train_epoch', type=int, default=5)
parser.add_argument('--lambda_ct', type=int, default=20)
parser.add_argument('--model_saving_path', type=str, default="./trained_model/dual/")
parser.add_argument('--pretrained_lightcnn_path', type=str,
                    default="./pretrained_model/LightCNN_29Layers_V2_checkpoint.pth.tar")
args = parser.parse_args()


def pre_train(train_loader, model, optimizer, epoch):
    i = 0
    top1 = AverageMeter()
    top5 = AverageMeter()

    for vis, inf, l in tqdm(train_loader):
        i += 1
        l = l.to(device)

        vis = vis.to(device)
        inf = inf.to(device)
        batch_size = vis.shape[0]

        logits_vis, logits_inf, _, _ = model(vis, inf, l, 0)

        loss = cri(logits_vis, l) + cri(logits_inf, l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #measure accuracy and record loss
        prec1, prec5 = accuracy(logits_inf.data, l.data, topk=(1, 5))
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        # print log
        if i % 10 == 0:
            info = "====> Epoch: [{:0>3d}][{:3d}/{:3d}] | ".format(epoch, i, len(train_loader))
            info += "Loss: {:4.3f}".format(loss.item())
            info += "Prec@1: {:4.2f} ({:4.2f}) Prec@5: {:4.2f} ({:4.2f})".format(top1.val, top1.avg, top5.val, top5.avg)
            print(info)


def train(train_loader, model, optimizer, epoch):
    i = 0
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    for vis, inf, l in tqdm(train_loader):
        i += 1
        l = l.to(device)

        vis = vis.to(device)
        inf = inf.to(device)
        batch_size = vis.shape[0]

        logits_vis, logits_inf, fc_vis, fc_inf = model(vis, inf, l, 0)

        loss_ct = contrastive_loss(fc_inf, fc_vis, l.detach().cpu().numpy())
        loss_ce = cri(logits_vis, l) + cri(logits_inf, l)

        loss = loss_ce + args.lambda_ct * loss_ct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(logits_inf.data, l.data, topk=(1, 5))
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        # print log
        if i % 10 == 0:
            info = "====> Epoch: [{:0>3d}][{:3d}/{:3d}] | ".format(epoch, i, len(train_loader))
            info += "Loss: ce: {:4.3f} ct: {:4.3f}| ".format(loss_ce.item(), loss_ct.item())
            info += "Prec@1: {:4.2f} ({:4.2f}) Prec@5: {:4.2f} ({:4.2f})".format(top1.val, top1.avg, top5.val, top5.avg)
            print(info)


for _ in range(1):
    if not os.path.exists(args.model_saving_path):
        os.makedirs(args.model_saving_path)
        print(f"folder is created: {args.model_saving_path}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset, num_classes = make_train_dataset(args.vis_train_text, args.inf_train_text, args.gen_vis_path,
                                                    args.vis_root, args.inf_root)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    cri = nn.CrossEntropyLoss()

    model = create_model(args.model_type, args.pretrained_lightcnn_path, device, num_classes + 1).to(device)

    '''
    Stage I: model pretrained for only cross entropy loss
    '''
    optimizer_pretrain = torch.optim.SGD(model.parameters(),  lr=args.lr, momentum=0.9,
                                         weight_decay=1e-4)
    for epoch in range(args.pretrain_epoch):

        pre_train(train_loader, model, optimizer_pretrain, epoch)
        torch.save(model.state_dict(), args.model_saving_path + args.model_type + "_pretrain.pt")

    '''
    Stage II: model finetune for cross entropy loss + contrastive_loss
    '''
    optimizer = torch.optim.SGD(model.parameters(),  lr=args.lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.train_epoch):
        adjust_learning_rate(args.lr, 2, optimizer, epoch)
        train(train_loader, model, optimizer, epoch)
        torch.save(model.state_dict(), args.model_saving_path + args.model_type + "_train_" + str(epoch) + ".pt")

