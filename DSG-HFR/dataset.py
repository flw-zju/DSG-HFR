from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import datasets, transforms
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, vis, inf, label):
        self.vis = vis
        self.inf = inf
        self.label = label
        self.trans = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])

    def __getitem__(self, item):

        vis = self.trans(Image.open(self.vis[item]).convert('L'))
        start = item - 20 if item > 20 else 0
        end = item + 20 if item + 20 < len(self.inf) else len(self.inf)
        idx = [j for j in range(start, end) if self.label[j] == self.label[item]]
        n_ = np.random.randint(0, len(idx))
        inf = self.trans(Image.open(self.inf[idx[n_]]).convert('L'))

        return vis, inf, self.label[item]

    def __len__(self):
        return len(self.vis)


def make_train_dataset(vis_txt, inf_txt, path, vis_root, inf_root):
    vis_dic = []
    inf_dic = []
    vis_c = []

    pre = -1
    zjr = -1
    with open(vis_txt, 'r') as f:
        for line in f:
            image_path = line.split('\n')[0]
            image_c = image_path.split('-')[0]
            image_path = os.path.join(vis_root, image_path)
            if os.path.exists(image_path):
                if image_c != pre:
                    pre = image_c
                    zjr += 1
                vis_dic.append(image_path)
                vis_c.append(zjr)
    f.close()

    pre = -1
    zjr = -1
    with open(inf_txt, 'r') as f:
        for line in f:
            image_path = line.split('\n')[0]
            image_c = image_path.split('-')[0]
            image_path = os.path.join(inf_root, image_path)
            if os.path.exists(image_path):
                if image_c != pre:
                    pre = image_c
                    zjr += 1
                inf_dic.append(image_path)
    f.close()
    zjr += 1

    path_label = datasets.ImageFolder(path)
    for i in range(len(path_label)):

        image_path = path_label.imgs[i][0]

        image_label = int(image_path.split('/')[-1].split('-')[0])

        vis_dic.append(image_path)
        vis_c.append(image_label + zjr)

        inf_path = image_path.replace('vis', 'inf')
        inf_dic.append(inf_path)

    print("")
    print("The numbers of all kinds of images are:")
    print(f"Vis {len(vis_dic)}, Inf {len(inf_dic)}")

    dataset = TrainDataset(vis_dic, inf_dic, vis_c)
    return dataset, max(vis_c)


def make_eval_dataset(vis_eval_txt, inf_eval_txt, vis_root, inf_root, vis_train_txt):
    vis_dic = []
    inf_dic = []
    vis_c = []
    inf_c = []

    with open(vis_eval_txt, 'r') as f:
        for line in f:
            path = line.split('\n')[0]
            image_path = os.path.join(vis_root, path)
            if os.path.exists(image_path):
                l = int(path.split('/')[-1].split('-')[0])
                vis_dic.append(image_path)
                vis_c.append(l)
    f.close()

    with open(vis_train_txt, 'r') as f:
        for line in f:
            path = line.split('\n')[0]
            image_path = os.path.join(vis_root, path)
            if os.path.exists(image_path):
                l = int(path.split('/')[-1].split('-')[0])
                vis_dic.append(image_path)
                vis_c.append(l)
    f.close()

    with open(inf_eval_txt, 'r') as f:
        for line in f:
            path = line.split('\n')[0]
            image_path = os.path.join(inf_root, path)
            if os.path.exists(image_path):
                l = int(path.split('/')[-1].split('-')[0])
                inf_dic.append(image_path)
                inf_c.append(l)
    f.close()

    print(f"Vis {len(vis_dic)}, Inf {len(inf_dic)}")
    print('................')
    vis = [(p, l) for (p, l) in zip(vis_dic, vis_c)]
    inf = [(p, l) for (p, l) in zip(inf_dic, inf_c)]
    return vis, inf