import numpy as np
import argparse
from tqdm import tqdm
import yaml
from attrdict import AttrMap
import cv2
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_manager import TestDataset
from utils import gpu_manage, save_image, heatmap
from SpA_Former import Generator


def predict(config, args):
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')  # 使用 GPU
        print("Using GPU")
    else:
        device = torch.device('cpu')  # 强制使用 CPU
        print("Using CPU")

    dataset = TestDataset(args.test_dir, config.in_ch, config.out_ch)
    data_loader = DataLoader(dataset=dataset, num_workers=config.threads, batch_size=1, shuffle=False)

    ### MODELS LOAD ###.......
    print('===> Loading models')

    gen = Generator(gpu_ids=[])

    param = torch.load(args.pretrained, map_location=device)
    gen.load_state_dict(param)
    gen = gen.to(device)


    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            x = Variable(batch[0])
            filename = batch[1][0]
            x = x.to(device)
            att , out = gen(x)
            h = 1
            w = 3
            c = 3
            p = config.width
            q = config.height
            allim = np.zeros(( h, w, c, p, q))
            x_ = x.cpu().numpy()[0]
            out_ = out.cpu().numpy()[0]
            in_rgb = x_[:3]
            out_rgb = np.clip(out_[:3], 0, 1)
            att_ = att.cpu().numpy()[0] * 255
            heat_att = heatmap(att_.astype('uint8'))
            allim[0, 0, :] = in_rgb * 255
            allim[0, 1, :] = out_rgb * 255
            allim[0, 2, :] = heat_att
            allim = allim.transpose(0, 3, 1, 4, 2)
            allim = allim.reshape((h*p, w*q, c))
            # out_rgb = np.transpose(out_rgb, (1, 2, 0))  # 转为 [height, width, 3]
            # out_rgb_resized = cv2.resize(out_rgb, (640, 480))  # 使用 OpenCV 调整大小
            # out_rgb_resized = (out_rgb_resized * 255).astype(np.uint8)   #添加的
            # save_image(args.out_dir, out_rgb_resized, i, 1, filename=filename) #添加的 只要一列去除阴影的图像
            save_image(args.out_dir, allim , i, 1, filename=filename)
if __name__ == '__main__':


    class Args:
        # 设置参数默认值
        config = 'config.yml'
        test_dir = ''
        out_dir = ''
        pretrained = ''
        cuda = False
        gpu_ids = []
        manualSeed = 0


    args = Args()
    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    config = AttrMap(config)
    predict(config, args)
