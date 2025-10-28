import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_curve
from dataset import make_eval_dataset
from lightcnn import LightCNN_29v2_cosface
from dual_tower import create_model
from utils import load_weights
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--vis_train_text', type=str, default="/home/project_texts/vis_train.txt")
parser.add_argument('--vis_test_text', type=str, default="/home/project_texts/vis_gallery.txt")
parser.add_argument('--inf_probe_text', type=str, default="/home/project_texts/inf_probe.txt")
parser.add_argument('--vis_root', type=str, default="/home/tufts_dataset/vis/data/")
parser.add_argument('--inf_root', type=str, default="/home/tufts_dataset/inf/data/")
parser.add_argument('--model_type', type=str, default="lightcnn",
                    help="lightcnn for LightCNN-29, "
                         "dual for DualTowerModel or "
                         "DualTowerTransformer for DualTowerTransformer")
parser.add_argument('--model_path', type=str, default="./trained_model/lightcnn/")
parser.add_argument('--pretrained_lightcnn_path', type=str,
                    default="./pretrained_model/LightCNN_29Layers_V2_checkpoint.pth.tar")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


trans = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def get_feats_lightcnn(data_list, model):
    imgs = []
    labels = []
    with torch.no_grad():
        for (path, label) in data_list:
            model.eval()
            img = trans(Image.open(path).convert('L')).unsqueeze(0).to(device)
            feat = model(img).squeeze()
            imgs.append(feat.cpu().numpy())
            labels.append(label)
    return np.array(imgs).astype(np.float32), np.array(labels)


def get_feats_dual(data_list, model, mode="vis"):
    imgs = []
    labels = []
    with torch.no_grad():
        for (path, label) in data_list:
            model.eval()
            img = trans(Image.open(path).convert('L')).unsqueeze(0).to(device)
            feat = model.eval_(img, mode).squeeze()
            imgs.append(feat.cpu().numpy())
            labels.append(label)
    return np.array(imgs).astype(np.float32), np.array(labels)


def eval(vis_f, nir_f, label, fars=[1e-3, 1e-2]):
    query_num = nir_f.shape[0]

    similarity = np.abs(np.dot(nir_f, vis_f.T))
    # print(similarity[0,...])

    top_inds = np.argsort(-similarity)

    label = label.T

    # calculate top1
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0]
        if label[i, j] == 1:
            correct_num += 1

    top1 = correct_num / query_num
    print("top1 = {:.2%}".format(top1))

    labels_ = label.flatten()

    similarity_ = similarity.flatten()

    fpr, tpr, thr = roc_curve(labels_, similarity_)

    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)

    tpr_fpr_row = []

    for far in fars:
        _, min_index = min(list(zip(abs(fpr - far), range(len(fpr)))))
        tpr_fpr_row.append(tpr[min_index])

        print("TPR {:.2%} @ FAR {:.4%}".format(tpr[min_index], far), thr[min_index])

    return top1, tpr_fpr_row


def Eval(model, model_type="lightcnn"):
    acc_l = []
    tar1 = []
    tar2 = []
    print("Start ! ***************************************************")
    vis, nir = make_eval_dataset(args.vis_test_text, args.inf_probe_text, args.vis_root, args.inf_root, args.vis_train_text)
    if model_type == "lightcnn":
        vis_f, vis_l = get_feats_lightcnn(vis, model)
        nir_f, nir_l = get_feats_lightcnn(nir, model)
    else:
        vis_f, vis_l = get_feats_dual(vis, model, "vis")
        nir_f, nir_l = get_feats_dual(nir, model, "inf")

    l = np.equal.outer(vis_l, nir_l).astype(np.float32)

    acc, tarfar = eval(vis_f, nir_f, l)
    acc_l.append(acc)
    tar1.append(tarfar[0])
    tar2.append(tarfar[1])
    print("End   ! ***************************************************")
    return acc_l, tar1, tar2


if __name__ == '__main__':
    if args.model_type == "lightcnn":
        model = LightCNN_29v2_cosface(args.pretrained_lightcnn_path, device, 1)
    else:
        model = create_model(args.model_type, args.pretrained_lightcnn_path, device, 1)
    model = load_weights(model, args.model_path, device)
    acc_l, tar1, tar2 = Eval(model, args.model_type)








