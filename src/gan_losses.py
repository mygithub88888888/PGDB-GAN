import torch
import torch.nn as nn
import torch.nn.functional as F

class TextureGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg = self._load_vgg()

    def _load_vgg(self):
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        vgg = vgg.features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg

    def perceptual_loss(self, fake, real):
        fake_feat = self.vgg(fake)
        real_feat = self.vgg(real.detach())
        return self.l1_loss(fake_feat, real_feat)

    def adversarial_loss(self, disc_outputs, is_real):
        losses = []
        for output in disc_outputs:
            target = torch.ones_like(output) if is_real else torch.zeros_like(output)
            losses.append(F.binary_cross_entropy_with_logits(output, target))
        return sum(losses) / len(losses)

    def forward(self, gen_output, real_img, disc_fake, disc_real):
        adv_loss = self.adversarial_loss(disc_fake, False)

        percep_loss = self.perceptual_loss(gen_output, real_img)

        recon_loss = self.l1_loss(gen_output, real_img)

        return {
            'total': adv_loss + 0.1 * percep_loss + 0.9 * recon_loss,
            'adv': adv_loss,
            'percep': percep_loss,
            'recon': recon_loss
        }