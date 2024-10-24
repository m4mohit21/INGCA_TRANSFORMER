from typing import Any
from thinc import config
import torch
from VIT import *
from models.model import BINMODEL
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from einops import rearrange
import loadData2 as loadData
import utils as utils
import os

# Configuration from external config file
cfg = Configs().parse()

FLIPPED = False
THRESHOLD = 0.5

SPLITSIZE = cfg.split_size
SETTING = cfg.vit_model_size
TPS = cfg.vit_patch_size

batch_size = cfg.batch_size
patch_size = TPS
image_size = (SPLITSIZE, SPLITSIZE)
MASKINGRATIO = 0.5

TEST_DATASET_PATH = cfg.testing_dataset  # Dataset directory for testing
PREDICTION_SAVE_PATH = cfg.prediction_save_path  # Where to save predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utility functions
imvisualize = utils.imvisualize
load_data_func = loadData.loadData_sets

# Function to load test data
def test_data_loader():
    _, _, data_test = load_data_func(flipped=FLIPPED)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return test_loader

# Initialize model
def init_model():
    if SETTING == 'base':
        ENCODERLAYERS = 6
        ENCODERHEADS = 8
        ENCODERDIM = 768
    else:
        raise ValueError(f"Model setting {SETTING} is not supported.")
    
    v = ViT(
        image_size=256,
        patch_size=patch_size,
        num_classes=1000,
        dim=ENCODERDIM,
        depth=ENCODERLAYERS,
        heads=ENCODERHEADS,
        mlp_dim=2048
    )

    IN = ViT(
        image_size=256,
        patch_size=patch_size,
        num_classes=1000,
        dim=768,
        depth=4,
        heads=6,
        mlp_dim=2048
    )

    model = BINMODEL(
        encoder=v,
        inner_encoder=IN,
        masking_ratio=MASKINGRATIO,
        decoder_dim=ENCODERDIM,
        decoder_depth=ENCODERLAYERS,
        decoder_heads=ENCODERHEADS
    )
    return model

# Load model weights
def load_model_weights(model, model_weights_path):
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    return model

# Function to save predictions
def save_predictions(predictions, filenames, epoch):
    os.makedirs(PREDICTION_SAVE_PATH, exist_ok=True)
    for i, pred in enumerate(predictions):
        save_path = os.path.join(PREDICTION_SAVE_PATH, f'prediction_{filenames[i]}_epoch{epoch}.png')
        utils.save_image(pred, save_path)

# Visualization function without ground truth
def visualize_predictions(epoch, model, test_loader):
    model.eval()
    all_preds = []
    filenames = []

    with torch.no_grad():
        for i, (test_index, test_in, _) in enumerate(test_loader):
            inputs = test_in.to(device)
            bs = len(test_in)

            _, pred_pixel_values = model(inputs, inputs)  # Auto-encoder style forward pass
            rec_patches = pred_pixel_values
            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                   p1=patch_size, p2=patch_size, h=image_size[0] // patch_size)

            for j in range(bs):
                all_preds.append(rec_images[j].cpu())
                filenames.append(f'test_image_{test_index[j]}')

            if i % 10 == 0:
                print(f'Processed batch {i+1}/{len(test_loader)}')

    # Save predicted images
    save_predictions(all_preds, filenames, epoch)

# Main test function
def test_model():
    model = init_model()
    model = load_model_weights(model, cfg.model_weights_path)
    
    test_loader = test_data_loader()
    
    print("Starting prediction...")
    visualize_predictions(0, model, test_loader)  # Use epoch 0 for simplicity

if __name__ == '__main__':
    test_model()
