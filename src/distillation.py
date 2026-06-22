import torch
import torch.nn as nn
import torch.nn.functional as F
from model111 import Network, Discriminator
from read111 import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import os

checkpoint_dir = './GAN/face_aware_checkpoints'
visualization_dir = './GAN/face_aware_visualizations'
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(visualization_dir, exist_ok=True)

def get_gabor_filters(kernel_size=3, sigma=1.0, theta=0, Lambda=1.0, psi=0, gamma=0.5):
    y, x = np.ogrid[-kernel_size // 2 + 1:kernel_size // 2 + 1, -kernel_size // 2 + 1:kernel_size // 2 + 1]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-(x_theta **2 + gamma** 2 * y_theta** 2) / (2 * sigma** 2)) * np.cos(
        2 * np.pi * x_theta / Lambda + psi)
    return gb

class GaborNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, num_directions=6):
        super(GaborNetwork, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.num_directions = num_directions

        frequency_bands = {
            'low': 5.0,
            'mid': 2.0,
        }

        for freq_name, freq_Lambda in frequency_bands.items():
            for i in range(num_directions):
                theta = i * np.pi / num_directions
                gabor_filter = get_gabor_filters(
                    kernel_size=kernel_size,
                    theta=theta,
                    Lambda=freq_Lambda
                )
                gabor_filter = torch.from_numpy(gabor_filter).float().unsqueeze(0).unsqueeze(0)
                gabor_filter = gabor_filter.repeat(out_channels, in_channels, 1, 1)

                conv_layer = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    bias=False
                )
                conv_layer.weight.data = gabor_filter
                conv_layer.weight.requires_grad = False
                self.conv_layers.append(conv_layer)

        self.freq_weights = nn.Parameter(torch.tensor([1.0, 1.5]))

    def forward(self, x):
        outputs = []
        band_outputs = []

        for i, conv_layer in enumerate(self.conv_layers):
            band_idx = i // self.num_directions
            if i % self.num_directions == 0:
                band_outputs.append([])
            band_outputs[band_idx].append(conv_layer(x))

        for band_idx, band_features in enumerate(band_outputs):
            band_tensor = torch.cat(band_features, dim=1)
            outputs.append(band_tensor * self.freq_weights[band_idx])

        output = torch.cat(outputs, dim=1)
        return output

def face_aware_distillation_loss(teacher_output, student_output, face_mask, gabor_teacher, gabor_student, temperature=1.0):
    """
    Gabor
    """
    teacher_feature = teacher_output[0]
    student_feature = student_output[0]
    mask = F.interpolate(face_mask.unsqueeze(1).float(), size=teacher_feature.shape[2:], mode='bilinear').squeeze(1)
    masked_teacher = teacher_feature * mask.unsqueeze(1)
    masked_student = student_feature * mask.unsqueeze(1)
    feature_loss = nn.MSELoss()(masked_student, masked_teacher)

    gabor_teacher_feature = gabor_teacher
    gabor_student_feature = gabor_student
    gabor_mask = F.interpolate(face_mask.unsqueeze(1).float(), size=gabor_teacher_feature.shape[2:], mode='bilinear').squeeze(1)
    masked_gabor_teacher = gabor_teacher_feature * gabor_mask.unsqueeze(1)
    masked_gabor_student = gabor_student_feature * gabor_mask.unsqueeze(1)
    gabor_loss = nn.MSELoss()(masked_gabor_student, masked_gabor_teacher)

    total_loss = feature_loss + 0.5 * gabor_loss
    return total_loss

def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1)
    mu2 = F.avg_pool2d(img2, window_size, stride=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.mean(ssim_map)

def plot_training_curves(losses, psnrs, ssims, save_path):
    """"""
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(psnrs, 'g-', linewidth=2)
    plt.title('Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(ssims, 'r-', linewidth=2)
    plt.title('Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_and_evaluate(student_model, gabor_model, dataloader, epoch=None):
    student_model.eval()
    gabor_model.eval()
    mse_scores = []
    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for i, (img, _, mask) in enumerate(dataloader):
            if i >= 5:
                break

            img = img.cuda()
            mask = mask.cuda()

            student_output = student_model(img)
            output_img = student_output[0] if isinstance(student_output, tuple) else student_output
            output_img = torch.clamp(output_img, 0, 1)
            if output_img.shape != img.shape:
                output_img = F.interpolate(output_img, size=img.shape[2:], mode='bilinear', align_corners=False)

            gabor_output = gabor_model(img)

            if i == 0:
                plt.figure(figsize=(18, 5))
                plt.subplot(1, 4, 1)
                plt.imshow(img[0].permute(1, 2, 0).cpu().numpy())
                plt.title('Input Image')
                plt.axis('off')

                plt.subplot(1, 4, 2)
                plt.imshow(mask[0].cpu().numpy(), cmap='gray')
                plt.title('Face Mask')
                plt.axis('off')

                plt.subplot(1, 4, 3)
                plt.imshow(output_img[0].permute(1, 2, 0).cpu().numpy())
                plt.title('Student Output')
                plt.axis('off')

                plt.subplot(1, 4, 4)
                plt.imshow(gabor_output[0, 0].cpu().numpy(), cmap='jet')
                plt.title('Gabor Feature')
                plt.axis('off')

                if epoch is not None:
                    plt.suptitle(f'Epoch {epoch} Visualization')
                    plt.savefig(f'{visualization_dir}/epoch_{epoch}.png')
                else:
                    plt.suptitle('Final Visualization')
                    plt.savefig(f'{visualization_dir}/final_result.png')
                plt.close()

            mse = mean_squared_error(img[0].cpu().numpy().flatten(), output_img[0].cpu().detach().numpy().flatten())
            psnr = calculate_psnr(img, output_img)
            ssim = calculate_ssim(img, output_img)

            mse_scores.append(mse)
            psnr_scores.append(psnr.item())
            ssim_scores.append(ssim.item())

            del img, mask, student_output, output_img, gabor_output
            torch.cuda.empty_cache()

    if mse_scores:
        avg_mse = sum(mse_scores) / len(mse_scores)
        avg_psnr = sum(psnr_scores) / len(psnr_scores)
        avg_ssim = sum(ssim_scores) / len(ssim_scores)

        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")

        return avg_mse, avg_psnr, avg_ssim
    else:
        print("No samples to evaluate!")
        return 0, 0, 0

def main():
    teacher_model = Network().cuda()
    teacher_weights_path = './train_results/LOL_results/Train-20250518-205252/model_epochs/weights_4500.pt'
    try:
        teacher_state_dict = torch.load(teacher_weights_path, map_location='cuda', weights_only=True)
        teacher_model.load_state_dict(teacher_state_dict)
        print(f" : {teacher_weights_path}")
    except Exception as e:
        print(f" : {e}")
        return

    student_model = Network().cuda()
    gabor_model = GaborNetwork().cuda()
    teacher_gabor = GaborNetwork().cuda()

    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    teacher_gabor.eval()

    dataloader = DataLoader(
        img_dir='./datasets/JIAGAN/image',
        mask_dir='./datasets/JIAGAN/mask',
        task='train'
    )
    dataloader = torch.utils.data.DataLoader(dataloader, batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    best_psnr = float('-inf')
    lambda_distill = 0.7
    lambda_gan = 0.1
    lambda_gabor = 0.5

    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_psnrs = []
    val_ssims = []

    for epoch in range(5):
        student_model.train()
        epoch_loss = 0.0

        for i, (img, _, face_mask) in enumerate(dataloader):
            img = img.cuda()
            face_mask = face_mask.cuda()

            with torch.no_grad():
                teacher_output = teacher_model(img)
                teacher_gabor_output = teacher_gabor(img)

            with torch.cuda.amp.autocast():
                student_output = student_model(img)
                student_gabor_output = gabor_model(img)

                distill_loss = face_aware_distillation_loss(
                    teacher_output, student_output, face_mask,
                    teacher_gabor_output, student_gabor_output
                )

                discriminator = Discriminator().cuda()
                real_validity = discriminator(img)
                fake_validity = discriminator(student_output[0])
                gan_loss = -torch.mean(fake_validity)

                total_loss = lambda_distill * distill_loss + lambda_gan * gan_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()

            del img, face_mask, teacher_output, student_output, teacher_gabor_output, student_gabor_output
            del distill_loss, gan_loss, total_loss, real_validity, fake_validity
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/10], Loss: {avg_loss:.4f}")

        train_losses.append(avg_loss)

        current_mse, current_psnr, current_ssim = visualize_and_evaluate(student_model, gabor_model, dataloader, epoch)

        val_psnrs.append(current_psnr)
        val_ssims.append(current_ssim)

        plot_training_curves(
            train_losses,
            val_psnrs,
            val_ssims,
            f"{visualization_dir}/training_curves.png"
        )

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            torch.save(
                student_model.state_dict(),
                f"{checkpoint_dir}/best_student_model.pth"
            )
            print(f" PSNR: {best_psnr:.4f} dB")

        torch.save(
            student_model.state_dict(),
            f"{checkpoint_dir}/epoch_{epoch}.pth"
        )

    print("=====  =====")
    visualize_and_evaluate(student_model, gabor_model, dataloader)

if __name__ == "__main__":
    main()