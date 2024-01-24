import os
import torch

# Saves a model to file, and names it after the current epoch
def save_checkpoint(model, epoch, save_dir):
    filename = f"checkpoint_{epoch}.pth"
    save_path = f"{save_dir}/{filename}"
    torch.save(model.state_dict(),save_path)


def ResNet_save_checkpoint(model, epoch, save_dir):
    filename = f"ResNet_checkpoint_{epoch}.pt"
    save_path = os.path.join(save_dir, filename)
    torch.save(model, save_path)
    
    
def Final_ResNet_save_checkpoint(model, epoch, save_dir):
    filename = f"Final_ResNet_checkpoint_{epoch}.pt"
    save_path = os.path.join(save_dir, filename)
    torch.save(model, save_path)