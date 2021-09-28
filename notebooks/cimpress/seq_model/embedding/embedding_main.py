"""
    The main file of encoder-decoder training and save embedding.
    Here we use Pytorch's CIFAR-10 image as example for testing the autoencoder.

"""

import torch as th
import torchvision
from torchvision.datasets import cifar
import torchvision.transforms as transforms
import argparse

from models.embed_model import ConvEncoderDecoder
from torch.utils.data import DataLoader

import sys
import os
# add parent directory to path to load module from it
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from dataset.rawdata import ImageDataset


def train_transformings():
    transformings = transforms.Compose([
        transforms.Pad(int((100-32)/2)),
        transforms.ToTensor()
    ])
    return transformings

def raw_transformings():
    transformings = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((100, 100))
    ])
    return transformings

def train(args):
    # 0. Mis part
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print("Use device {}".format(device))

    # 1. Process dataset
    if args.data_set == 'cifar':
        data = torchvision.datasets.CIFAR10(root='../data', download=True, transform=train_transformings())
    elif args.data_set =='raw':
        data = ImageDataset(raw_transformings(), '../data/asset/art', '../data/asset/shading')
    else:
        raise Exception('Only support cifar and raw so far...')
    # output dataset information
    print(data)
    
    dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    # 2. define the autoencoder
    convcoder = ConvEncoderDecoder()
    convcoder = convcoder.to(device)

    # 3. define loss function and optimizer
    loss_fn = th.nn.MSELoss().to(device)
    optim = th.optim.Adam(convcoder.parameters(), lr=args.lr)

    # 4. train epoch loop
    for epoch in range(args.epochs):
        for i, imgs in enumerate(dataloader):
            if args.data_set == 'cifar':
                imgs, labels = imgs
            convcoder.train()
            imgs = imgs.to(device)

            reconst = convcoder(imgs)

            tr_loss = loss_fn(reconst, imgs)

            optim.zero_grad()
            tr_loss.backward()
            optim.step()

        print("Run {:03d} epoch, loss {:.6f}".format(epoch, tr_loss.item()))
    
    # 5. save encoder for inference
    th.save(convcoder.state_dict(), f'output/convcoder_{args.data_set}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AutoEncoder Test with CIFAR-10 images.")
    parser.add_argument("--data-set", type=str, default="cifar")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-worker", type=int, default=8)

    args = parser.parse_args()
    print(args)

    train(args)
