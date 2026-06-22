import os
import sys
import time
import glob
import numpy as np
import utilspro
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model111 import *
from read111 import DataLoader

parser = argparse.ArgumentParser("ZERO-IG")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--epochs', type=int, default=5001)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--save', type=str,
                   default='./train_results/LOL_results')
parser.add_argument('--model_pretrain', type=str, default='')
parser.add_argument('--mask_dir', type=str,
                   default='./datasets/JIAGAN/mask')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utilspro.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = os.path.join(args.save, 'model_epochs')
image_path = os.path.join(args.save, 'image_epochs')
os.makedirs(model_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                   format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'train_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info(f'gpu device = {args.gpu}')
    logging.info(f"args = {args}")

    model = Network()
    utilspro.save(model, os.path.join(args.save, 'initial_weights.pt'))
    model = model.cuda()

    optimizer = torch.optim.Adam(model.enhance.parameters(), lr=args.lr,
                                betas=(0.9, 0.999), weight_decay=3e-4)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(),
                                  lr=args.lr * 0.1)

    train_img_dir = './datasets/JIAGAN/image'
    TrainDataset = DataLoader(img_dir=train_img_dir, mask_dir=args.mask_dir, task='train')
    test_img_dir = './datasets/JIAGAN/image'
    TestDataset = DataLoader(img_dir=test_img_dir, mask_dir=args.mask_dir, task='test')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=0,
        shuffle=False
    )

    test_queue = torch.utils.data.DataLoader(
        TestDataset,
        batch_size=1,
        pin_memory=True,
        num_workers=0,
        shuffle=False
    )

    total_step = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []

        for idx, (input, img_name, face_mask) in enumerate(train_queue):
            total_step += 1
            input = Variable(input.cuda(), requires_grad=False)
            face_mask = face_mask.cuda()

            optimizer_d.zero_grad()
            d_real = model.discriminator(input)
            enhanced = model.enhance(input)
            d_fake = model.discriminator(enhanced.detach())

            d_loss_real = torch.nn.BCEWithLogitsLoss()(d_real, torch.ones_like(d_real))
            d_loss_fake = torch.nn.BCEWithLogitsLoss()(d_fake, torch.zeros_like(d_fake))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            optimizer.zero_grad()
            loss = model._loss(input, face_mask)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            logging.info(f'train-epoch {epoch:03d} {idx:03d} {loss:.4f}')

        avg_loss = np.average(losses)
        logging.info(f'train-epoch {epoch:03d} avg_loss {avg_loss:.4f}')
        utilspro.save(model, os.path.join(model_path, f'weights_{epoch}.pt'))

        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                for idx, (input, img_name, _) in enumerate(test_queue):
                    input = Variable(input.cuda())
                    image_name = img_name[0].split('/')[-1].split('.')[0]

                    enhanced = model.enhance(input)
                    os.makedirs(f"{args.save}/result/enhance", exist_ok=True)
                    Image.fromarray(save_images(enhanced)).save(
                        f"{args.save}/result/enhance/{image_name}_enhance_{epoch}.png")

if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()