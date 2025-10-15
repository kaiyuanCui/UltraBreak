from PIL import Image

from utils import get_filtered_cands, sample_control, token_gradients
from llava15_adapter import Llava15Adapter
from llava16_adapter import Llava16Adapter
from qwen2_adapter import Qwen2Adapter
from clip_adapter import CLIPAdapter
from glm_adapter import GlmAdapter
import torch.optim as optim

import torch
import torchvision.transforms as transforms
import pandas as pd
import time
import random

import os
import json
import pickle


def initialise_patch(device, patch_size, image_path=None):
    if image_path is not None:
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor()
        ])
        img = Image.open(image_path).convert("RGB")
        adv_patch = transform(img).to(device)
    else:
        adv_patch = torch.rand(3, patch_size, patch_size, device=device)

    adv_patch.requires_grad_()
    return adv_patch

def save_tensor_as_image(tensor: torch.Tensor, save_path: str) -> None:
    """
    Convert a tensor of shape [1, 3, H, W] with values in [0,1] to a PIL Image and save it.

    Args:
        tensor: torch.Tensor of shape [1, 3, H, W], values expected in [0,1]
        save_path: Path to save the output image (e.g. 'outputs/optimised_patch.jpg')
    """
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/constants.py

    #print(tensor.shape)  # actually 3,336,336 
    
    
    tensor = tensor.cpu().clone().detach()
    if tensor.dim() == 4:  # [1, C, H, W]
        tensor = tensor.squeeze(0)  # Remove batch dim → [C, H, W]

    tensor = (tensor * 255).to(torch.uint8)
    np_img = tensor.permute(1, 2, 0).numpy()  # Convert [C,H,W] to [H,W,C]
    image = Image.fromarray(np_img)
    image.save(save_path)


def total_variation(img):
        """
        img: [C, H, W] or [B, C, H, W]
        Returns scalar TV loss
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)  # add batch dim
    
        # differences along height (dim=2)
        h_variation = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
        # differences along width (dim=3)
        w_variation = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()

        return h_variation + w_variation
    

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    train_config = "./train_configs/safebench_diffusion_explicit_force_jailbroken_mode.csv"
    qwen_adapter = Qwen2Adapter("Qwen/Qwen2-VL-7B-Instruct", patch_only=False)

    # optionally optimise against multiple surrogates; final method only uses one
    ensemble = [qwen_adapter]#, clip_adapter] #, llava_adapter]
    
    exp_name =  'test'
    base_epoch = 0
    
    os.makedirs(f"outputs/{exp_name}/", exist_ok=True)
    target_df = pd.read_csv(train_config)
    target_df = target_df.fillna("")

    if base_epoch > 0:
        adv_patch = initialise_patch(device, 224, f'outputs/{exp_name}/{base_epoch}.png')
    else:
        adv_patch = initialise_patch(device, 224, None)

    custom_loss = True

    # Set up optimizer
    lr = 0.01
    optimizer = optim.Adam([adv_patch], lr=lr) # tune the learning rate

    # Training loop
    num_epochs = 5000 
    patience = 5000  # Number of epochs to wait before stopping
    best_loss = float('inf')
    best_img = None
    no_improve_count = 0

    tv_weight = 0.5
    l2_weight = 0.0
    
    losses = []

    for epoch in range(base_epoch, num_epochs):
        start_time = time.time()
        optimizer.zero_grad()   
        text_loss = 0
        target_losses =[]
        for model in ensemble:
            for index, row in target_df.iterrows():
                if index == 0 and epoch % 20 == 0:
                    print_probs = True
                else:
                    print_probs = False
                
                # recompute to save memory
                l2_loss = torch.mean((adv_patch-0.5) ** 2)
                tv_loss = total_variation(adv_patch)
            
                model_loss= model.compute_loss(row, adv_patch, '', custom_loss=custom_loss, print_probs=print_probs)
                loss =  model_loss + tv_loss * tv_weight + l2_loss * l2_weight
                loss.backward()
                text_loss += model_loss.item()
                target_losses.append(model_loss.item())

        # this is for display only
        total_loss = text_loss / (len(target_df) * len(ensemble)) + tv_loss * tv_weight + l2_loss * l2_weight
      
        optimizer.step()
        
        #print("Patch min:", adv_patch.min().item(), "Patch max:", adv_patch.max().item())   
        # Clamp the image to keep valid values
        adv_patch.data = torch.clamp(adv_patch.data, 0, 1)

      

        if epoch % 20 == 0:
            # check output periodically
            with torch.no_grad():
                print("saving..")
                save_path = f"outputs/{exp_name}/{epoch}.png"
                save_tensor_as_image(adv_patch, save_path)

                for model in ensemble:
                    print(model.generate(target_df.iloc[0]['text'], save_path))

            # record the losses periodically
            df = pd.DataFrame(losses) 
            df.to_csv(f"outputs/{exp_name}/losses.csv", index=False)

        
        # Store the best image if the loss improves
        if total_loss < best_loss:
            best_loss = total_loss
            best_img = adv_patch.clone().detach()  # Save a copy of the best benign image
            no_improve_count = 0  # Reset patience counter
        else:
            no_improve_count += 1

        end_time = time.time()
        epoch_duration = end_time - start_time
        # Print loss for monitoring
        print(f"Epoch {epoch}: Avg Loss = {total_loss}, Text Loss = {text_loss/ (len(target_df) * len(ensemble))}, TV Loss = {tv_loss * tv_weight}, L2 Loss = {l2_loss * l2_weight}, Time Spent = {epoch_duration}s")

        loss_dict = {
            'epoch': epoch,
            'total_loss': total_loss.item(),
            'text_loss': text_loss,
            'tv_loss': tv_loss.item(),
            'tv_weight': tv_weight,
            'l2_loss': l2_loss.item(),
            'l2_weight': l2_weight
        }

        losses.append(loss_dict)
        
        
        # Early stopping condition
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}. Reverting to best model state.")
            adv_patch.data = best_img.data  # Revert to best saved state
            break  # Exit training loop

    save_tensor_as_image(adv_patch, f"outputs/{exp_name}/adv_patch_{epoch}.png")
    
    return

if __name__ == "__main__":
    main()