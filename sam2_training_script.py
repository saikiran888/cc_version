#!/usr/bin/env python3
import os
# ─── GPU SETUP ────────────────────────────────────────────────────────────────
# Use both GPUs for maximum performance
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Optimize CUDA memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# Enable faster CUDA operations
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch
# Enable TF32 for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable cudnn benchmarking for faster training
torch.backends.cudnn.benchmark = True
# Enable cudnn deterministic mode for reproducibility
torch.backends.cudnn.deterministic = True

import cv2
import numpy as np
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pycocotools.coco import COCO
from segment_anything import sam_model_registry
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
CHECKPOINT       = "/home/sai/Desktop/sam2/sam_vit_h_4b8939.pth"
DATA_DIR         = "/cellchorus/MRCNNv350Train/images"
COCO_JSON        = "/cellchorus/MRCNNv350Train/output_coco.json"
BATCH_SIZE       = 8  # Increased for dual A6000 GPUs
NUM_EPOCHS       = 15
LR               = 1e-4
NUM_WORKERS      = 12  # Increased for better throughput
HIGH_RES_WEIGHT  = 0.5
MODEL_DIR        = "trained_models"  # Directory to store trained models
GRAD_ACCUM_STEPS = 2  # Gradient accumulation steps for better memory efficiency
#─────────────────────────────────────────────────────────────────────────────────

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Create TensorBoard writer
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('runs', f'SAM2_training_{current_time}')
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs will be saved to {log_dir}")

class CocoImageDataset(Dataset):
    def __init__(self, data_dir, coco_json):
        self.data_dir = data_dir
        self.coco     = COCO(coco_json)
        self.img_ids  = self.coco.getImgIds()
        print(f"Found {len(self.img_ids)} images in dataset")

    def pad_to_1024(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        ph, pw = 1024 - h, 1024 - w
        top, bottom = ph // 2, ph - ph // 2
        left, right = pw // 2, pw - pw // 2
        
        # Log padding details
        print(f"Original size: {h}x{w}")
        print(f"Padding: top={top}, bottom={bottom}, left={left}, right={right}")
        
        padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=0)
        print(f"Padded size: {padded.shape[:2]}")
        return padded

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info   = self.coco.imgs[img_id]
        path   = os.path.join(self.data_dir, info["file_name"])
        img    = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read {path}")
            
        print(f"\nProcessing image: {info['file_name']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.pad_to_1024(img).astype(np.float32) / 255.0

        # gather all masks for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)
        masks   = [
            self.pad_to_1024(self.coco.annToMask(ann).astype(np.float32))
            for ann in anns
        ]
        
        # Ensure we have at least one mask
        if not masks:
            masks = [np.zeros((1024, 1024), dtype=np.float32)]
            
        # Pad with zero masks to ensure consistent number of masks
        max_masks = 10  # Set a reasonable maximum number of masks
        while len(masks) < max_masks:
            masks.append(np.zeros((1024, 1024), dtype=np.float32))
            
        # Take only the first max_masks masks
        masks = masks[:max_masks]
        masks = np.stack(masks, axis=0)  # (M, H, W)

        # to tensors
        img_tensor   = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        masks_tensor = torch.from_numpy(masks)                # (M, H, W)
        return img_tensor, masks_tensor

def train_multi_mask_decoder(device: str):
    # 1) Load SAM & freeze everything but mask_decoder
    print("Loading SAM ViT-H…")
    sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT)
    for name, p in sam.named_parameters():
        if not name.startswith("mask_decoder"):
            p.requires_grad = False
    try:
        sam.image_encoder.enable_attention_slicing()
    except Exception:
        pass
    
    # Create two model instances for each GPU
    sam_gpu0 = sam.to('cuda:0')
    # Properly clone the model for GPU 1
    sam_gpu1 = sam_model_registry["vit_h"](checkpoint=CHECKPOINT)
    sam_gpu1.load_state_dict(sam_gpu0.state_dict())
    sam_gpu1 = sam_gpu1.to('cuda:1')
    
    # Freeze non-mask-decoder parameters in GPU 1 model
    for name, p in sam_gpu1.named_parameters():
        if not name.startswith("mask_decoder"):
            p.requires_grad = False
    try:
        sam_gpu1.image_encoder.enable_attention_slicing()
    except Exception:
        pass
    
    print("Models loaded on both GPUs")

    # 2) Optimizer, loss, scaler
    # Combine parameters from both models
    params = list(sam_gpu0.parameters()) + list(sam_gpu1.parameters())
    optimizer = Adam(filter(lambda p: p.requires_grad, params), lr=LR)
    loss_fn   = BCEWithLogitsLoss()
    scaler    = GradScaler()

    # 3) DataLoader
    ds     = CocoImageDataset(DATA_DIR, COCO_JSON)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=True)  # Keep workers alive between epochs
    print(f"→ {len(ds)} images, batch_size={BATCH_SIZE}")

    # 4) Training loop
    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        sam_gpu0.train()
        sam_gpu1.train()
        total_loss = 0.0
        total_low_res_loss = 0.0
        total_high_res_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at start of epoch

        for batch_idx, (imgs, gt_masks) in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")):
            # Split batch between GPUs
            half_batch = BATCH_SIZE // 2
            imgs_gpu0 = imgs[:half_batch].to('cuda:0', non_blocking=True)
            imgs_gpu1 = imgs[half_batch:].to('cuda:1', non_blocking=True)
            gt_masks_gpu0 = gt_masks[:half_batch].to('cuda:0', non_blocking=True)
            gt_masks_gpu1 = gt_masks[half_batch:].to('cuda:1', non_blocking=True)

            # Process masks in smaller batches
            B, M, H, W = gt_masks_gpu0.shape
            batch_loss = 0.0
            batch_low_res_loss = 0.0
            batch_high_res_loss = 0.0
            
            # Process masks in groups of 4 (SAM's internal limit)
            for i in range(0, M, 4):
                # Get current batch of masks for both GPUs
                curr_masks_gpu0 = gt_masks_gpu0[:, i:i+4]  # (B, min(4, M-i), H, W)
                curr_masks_gpu1 = gt_masks_gpu1[:, i:i+4]  # (B, min(4, M-i), H, W)
                
                # Resize masks to match SAM's internal dimensions
                masks_resized_gpu0 = F.interpolate(
                    curr_masks_gpu0, 
                    size=(4, 4),  # SAM's internal mask size
                    mode='bilinear',
                    align_corners=False
                )
                masks_resized_gpu1 = F.interpolate(
                    curr_masks_gpu1, 
                    size=(4, 4),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Process each image in the batch separately for both GPUs
                for b in range(half_batch):
                    with autocast(device_type='cuda', dtype=torch.float16):
                        # GPU 0 processing
                        with torch.no_grad():
                            embs_gpu0 = sam_gpu0.image_encoder(imgs_gpu0[b:b+1])
                            image_pe_gpu0 = sam_gpu0.prompt_encoder.get_dense_pe()
                        
                        masks_for_prompt_gpu0 = masks_resized_gpu0[b].unsqueeze(1)
                        
                        with torch.no_grad():
                            sparse_pe_gpu0, dense_pe_gpu0 = sam_gpu0.prompt_encoder(
                                points=None,
                                boxes=None,
                                masks=masks_for_prompt_gpu0
                            )
                        
                        out_masks_gpu0, _ = sam_gpu0.mask_decoder(
                            image_embeddings=embs_gpu0,
                            image_pe=image_pe_gpu0,
                            sparse_prompt_embeddings=sparse_pe_gpu0,
                            dense_prompt_embeddings=dense_pe_gpu0,
                            multimask_output=False
                        )
                        
                        # GPU 1 processing
                        with torch.no_grad():
                            embs_gpu1 = sam_gpu1.image_encoder(imgs_gpu1[b:b+1])
                            image_pe_gpu1 = sam_gpu1.prompt_encoder.get_dense_pe()
                        
                        masks_for_prompt_gpu1 = masks_resized_gpu1[b].unsqueeze(1)
                        
                        with torch.no_grad():
                            sparse_pe_gpu1, dense_pe_gpu1 = sam_gpu1.prompt_encoder(
                                points=None,
                                boxes=None,
                                masks=masks_for_prompt_gpu1
                            )
                        
                        out_masks_gpu1, _ = sam_gpu1.mask_decoder(
                            image_embeddings=embs_gpu1,
                            image_pe=image_pe_gpu1,
                            sparse_prompt_embeddings=sparse_pe_gpu1,
                            dense_prompt_embeddings=dense_pe_gpu1,
                            multimask_output=False
                        )

                        # Reshape output masks
                        out_masks_gpu0 = out_masks_gpu0.view(-1, *out_masks_gpu0.shape[-2:])
                        out_masks_gpu1 = out_masks_gpu1.view(-1, *out_masks_gpu1.shape[-2:])

                        # Calculate losses for both GPUs
                        target_low_gpu0 = F.interpolate(
                            curr_masks_gpu0[b:b+1], size=out_masks_gpu0.shape[-2:], 
                            mode='bilinear', align_corners=False
                        ).squeeze(0)
                        target_low_gpu1 = F.interpolate(
                            curr_masks_gpu1[b:b+1], size=out_masks_gpu1.shape[-2:], 
                            mode='bilinear', align_corners=False
                        ).squeeze(0)

                        loss_low_gpu0 = loss_fn(out_masks_gpu0, target_low_gpu0)
                        loss_low_gpu1 = loss_fn(out_masks_gpu1, target_low_gpu1)

                        pred_high_gpu0 = F.interpolate(
                            out_masks_gpu0.unsqueeze(0), size=curr_masks_gpu0[b:b+1].shape[-2:],
                            mode='bilinear', align_corners=False
                        ).squeeze(0)
                        pred_high_gpu1 = F.interpolate(
                            out_masks_gpu1.unsqueeze(0), size=curr_masks_gpu1[b:b+1].shape[-2:],
                            mode='bilinear', align_corners=False
                        ).squeeze(0)

                        loss_high_gpu0 = loss_fn(pred_high_gpu0, curr_masks_gpu0[b])
                        loss_high_gpu1 = loss_fn(pred_high_gpu1, curr_masks_gpu1[b])

                        # Move GPU 1 losses to GPU 0 before combining
                        loss_low_gpu1 = loss_low_gpu1.to('cuda:0')
                        loss_high_gpu1 = loss_high_gpu1.to('cuda:0')

                        # Calculate combined losses
                        low_res_loss = (loss_low_gpu0 + loss_low_gpu1) / 2
                        high_res_loss = (loss_high_gpu0 + loss_high_gpu1) / 2
                        curr_loss = (low_res_loss + HIGH_RES_WEIGHT * high_res_loss) / GRAD_ACCUM_STEPS

                        batch_loss += curr_loss
                        batch_low_res_loss += low_res_loss
                        batch_high_res_loss += high_res_loss

            # Average loss over all mask groups and batch items
            loss = batch_loss / (B * ((M + 3) // 4))  # Average over batch and mask groups
            low_res_loss = batch_low_res_loss / (B * ((M + 3) // 4))
            high_res_loss = batch_high_res_loss / (B * ((M + 3) // 4))

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            total_low_res_loss += low_res_loss.item() * GRAD_ACCUM_STEPS
            total_high_res_loss += high_res_loss.item() * GRAD_ACCUM_STEPS

            # Log metrics to TensorBoard
            global_step += 1
            writer.add_scalar('Loss/total', loss.item(), global_step)
            writer.add_scalar('Loss/low_res', low_res_loss.item(), global_step)
            writer.add_scalar('Loss/high_res', high_res_loss.item(), global_step)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)

            # Log sample predictions periodically
            if global_step % 100 == 0:
                # Log a sample prediction from GPU 0
                writer.add_images('Predictions/GPU0', 
                                torch.sigmoid(out_masks_gpu0[:1].unsqueeze(1)), 
                                global_step)
                writer.add_images('Ground_Truth/GPU0', 
                                target_low_gpu0[:1].unsqueeze(0), 
                                global_step)
                # Log a sample prediction from GPU 1
                writer.add_images('Predictions/GPU1', 
                                torch.sigmoid(out_masks_gpu1[:1].unsqueeze(1)), 
                                global_step)
                writer.add_images('Ground_Truth/GPU1', 
                                target_low_gpu1[:1].unsqueeze(0), 
                                global_step)

        avg_loss = total_loss / len(loader)
        avg_low_res_loss = total_low_res_loss / len(loader)
        avg_high_res_loss = total_high_res_loss / len(loader)
        
        # Log epoch metrics
        writer.add_scalar('Epoch/Loss/total', avg_loss, epoch)
        writer.add_scalar('Epoch/Loss/low_res', avg_low_res_loss, epoch)
        writer.add_scalar('Epoch/Loss/high_res', avg_high_res_loss, epoch)
        
        print(f"✓ Epoch {epoch} — Avg Loss: {avg_loss:.4f} (Low: {avg_low_res_loss:.4f}, High: {avg_high_res_loss:.4f})")

        # Save checkpoint every epoch
        ckpt = os.path.join(MODEL_DIR, f"sam2_multimask_epoch{epoch}.pth")
        torch.save({
            "epoch":      epoch,
            "state_dict": {
                "gpu0": sam_gpu0.state_dict(),
                "gpu1": sam_gpu1.state_dict()
            },
            "optimizer":  optimizer.state_dict(),
            "loss":       avg_loss
        }, ckpt)
        print(f"  ➜ Saved checkpoint {ckpt}")

    # Close TensorBoard writer
    writer.close()
    print("✅ Training complete!")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_multi_mask_decoder(device) 
