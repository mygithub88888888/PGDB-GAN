import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import MDAGenerator, MultiScaleDiscriminator
from dataset import FaceTextureDataset
from losses import TextureGANLoss
import config
import os
from tqdm import tqdm
import datetime

def train():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    gen = MDAGenerator().to(config.DEVICE)
    disc = MultiScaleDiscriminator().to(config.DEVICE)

    opt_gen = Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    criterion = TextureGANLoss().to(config.DEVICE)

    train_loader = FaceTextureDataset.create_dataloader(
        config.DATA_ROOT, 'train', config.BATCH_SIZE
    )

    for epoch in range(config.NUM_EPOCHS):
        gen.train()
        disc.train()

        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            real_imgs = batch['image'].to(config.DEVICE)
            masks = batch['mask'].to(config.DEVICE)

            opt_disc.zero_grad()

            with torch.no_grad():
                fake_imgs = gen(real_imgs, masks)

            disc_real = disc(real_imgs)
            disc_fake = disc(fake_imgs.detach())

            loss_disc_real = criterion.adversarial_loss(disc_real, True)
            loss_disc_fake = criterion.adversarial_loss(disc_fake, False)
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            loss_disc.backward()
            opt_disc.step()

            opt_gen.zero_grad()

            fake_imgs = gen(real_imgs, masks)
            disc_fake = disc(fake_imgs)

            loss_dict = criterion(fake_imgs, real_imgs, disc_fake, disc_real)
            loss_gen = loss_dict['total']

            loss_gen.backward()
            opt_gen.step()

            loop.set_postfix(
                epoch=epoch,
                loss_disc=loss_disc.item(),
                loss_gen=loss_gen.item(),
                recon_loss=loss_dict['recon'].item()
            )

        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            torch.save(gen.state_dict(),
                      f"{config.CHECKPOINT_DIR}/gen_epoch_{epoch+1}.pth")
            torch.save(disc.state_dict(),
                      f"{config.CHECKPOINT_DIR}/disc_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()