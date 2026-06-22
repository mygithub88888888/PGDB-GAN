import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from model111 import Network
from read111 import DataLoader
import copy

def gabor_pruning_strategy(module, name, amount):
    prune.l1_unstructured(module, name=name, amount=amount)

def structured_pruning(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            gabor_pruning_strategy(module, 'weight', amount)
            if module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=amount)

def evaluate_model(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for img, _, mask in dataloader:
            img = img.cuda()
            mask = mask.cuda()
            output = model(img)
            output = output[0]
            _, predicted = torch.max(output.data, 1)
            total += mask.size(0)
            correct += (predicted == mask).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def calculate_compression_rate(original_model, pruned_model):
    original_params = sum(p.numel() for p in original_model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    compression_rate = (1 - pruned_params / original_params) * 100
    return compression_rate

def main():
    model = Network().cuda()

    dataloader = DataLoader(img_dir='path/to/images', mask_dir='path/to/masks', task='train')
    dataloader = torch.utils.data.DataLoader(dataloader, batch_size=4, shuffle=True)

    original_accuracy = evaluate_model(model, dataloader)
    print(f"Original model accuracy: {original_accuracy}%")

    pruned_model = copy.deepcopy(model)

    structured_pruning(pruned_model, amount=0.2)

    pruned_accuracy = evaluate_model(pruned_model, dataloader)
    print(f"Pruned model accuracy: {pruned_accuracy}%")

    compression_rate = calculate_compression_rate(model, pruned_model)
    print(f"Compression rate: {compression_rate}%")

if __name__ == "__main__":
    main()