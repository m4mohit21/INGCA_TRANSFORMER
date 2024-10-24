
from typing import Any
from thinc import config
import torch
from VIT import *
#from vit_pytorch import ViT
# from models.binae import BINMODEL
from models.model import BINMODEL 
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from einops import rearrange
import loadData2 as loadData
import utils as utils
from  config import Configs
import os
from sklearn.metrics import precision_score, recall_score, f1_score


cfg = Configs().parse() #testin_dataset in configs file

FLIPPED = False
THRESHOLD = 0.5

SPLITSIZE = cfg.split_size
SETTING = cfg.vit_model_size
TPS = cfg.vit_patch_size

batch_size = cfg.batch_size

experiment = SETTING +'_'+ str(SPLITSIZE)+'_' + str(TPS)

patch_size = TPS
image_size =  (SPLITSIZE,SPLITSIZE)

MASKINGRATIO = 0.5
VIS_RESULTS = True
TEST_DIBCO = cfg.testing_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
count_psnr = utils.count_psnr
#
count_PRF  = utils.count_PRF
#
imvisualize = utils.imvisualize
load_data_func = loadData.loadData_sets


best_psnr = 0
best_epoch = 0


def sort_batch(batch):
    n_batch = len(batch)
    train_index = []
    train_in = []
    train_out = []
    for i in range(n_batch):
        idx, img, gt_img = batch[i]

        train_index.append(idx)
        train_in.append(img)
        train_out.append(gt_img)

    train_index = np.array(train_index)
    train_in = np.array(train_in, dtype='float32')
    train_out = np.array(train_out, dtype='float32')

    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)

    return train_index, train_in, train_out


def test_data_loader():
    _, _, data_test = load_data_func(flipped=FLIPPED)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return  test_loader

test_loader = test_data_loader()

patch_size = 16
word_size = 8

if SETTING == 'base':
    ENCODERLAYERS = 6
    ENCODERHEADS = 8
    ENCODERDIM = 768

v = ViT(
    image_size = 256,
    patch_size = patch_size,
    word_size = word_size,
    num_classes = 1000,
    dim = ENCODERDIM,
    depth = ENCODERLAYERS,
    heads = ENCODERHEADS,
    mlp_dim = 2048
)

IN = ViT(
    image_size = 256,
    patch_size = patch_size,
    word_size = word_size,
    num_classes = 1000,
    dim = 768,
    depth = 4,
    heads = 6,
    mlp_dim = 2048
)

MASKINGRATIO = 0.5
model = BINMODEL(
    encoder = v,
    inner_encoder = IN,
    masking_ratio = MASKINGRATIO,   ## __ doesnt matter for binarization
    decoder_dim = ENCODERDIM,
    decoder_depth = ENCODERLAYERS,
    decoder_heads = ENCODERHEADS       # anywhere from 1 to 
)


model = model.to(device)
optimizer = optim.AdamW(model.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)




def visualize(epoch):
    losses = 0
    all_preds = []
    all_targets = []
    for i, (test_index, test_in, test_out) in enumerate(test_loader):
        # inputs, labels = data
        bs = len(test_in)
        print(test_index)
        inputs = test_in.to(device)
        outputs = test_out.to(device)

        with torch.no_grad():
            loss, pred_pixel_values = model(inputs,outputs)
            
            rec_patches = pred_pixel_values

            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
            
            for j in range (0,bs):
                imvisualize(imdeg=inputs[j].cpu(),imgt =outputs[j].cpu(),impred =rec_images[j].cpu(),ind= test_index[j],epoch='0',setting = 'experiment')
                #imdeg, impred, ind, imgt='None', epoch='0', setting=''
            losses += loss.item()
    
    print('valid loss: ', losses / len(test_loader))
           
        # if i == 5:
        #     break            
            # losses += loss.item()
    #          # Convert the model's output to binary predictions using THRESHOLD
    #         preds = (rec_images > THRESHOLD).float()
    #         all_preds.extend(preds.cpu().numpy().flatten())
    #         all_targets.extend(outputs.cpu().numpy().flatten())
    
    #  # Calculate precision, recall, and F1-score
    # precision = precision_score(all_targets, all_preds, average='binary')
    # recall = recall_score(all_targets, all_preds, average='binary')
    # f1 = f1_score(all_targets, all_preds, average='binary')


    # print('valid loss: ', losses / len(test_loader))


    # print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    # print('Valid loss: ', losses / len(test_loader))


## also chage this  function for calling as above changes
# def valid_model(epoch):
    
#     psnr  = count_psnr(epoch,valid_data=TEST_DIBCO,setting=experiment,flipped=FLIPPED , thresh=THRESHOLD)
#     print('Test PSNR: ', psnr)

#     precision, recall, f1 = count_PRF(epoch,valid_data=TEST_DIBCO,setting=experiment,flipped=FLIPPED , thresh=THRESHOLD)
#     print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')


#new function
def valid_model(epoch):
    
    # Calculate PSNR for each image and the overall average PSNR
    psnr_values, avg_psnr = count_psnr(epoch, valid_data=TEST_DIBCO, setting=experiment, flipped=FLIPPED, thresh=THRESHOLD)
    print('Individual PSNR values: ', psnr_values)
    print('Overall Average PSNR: ', avg_psnr)
    
    # Calculate Precision, Recall, and F1-Score for each image and their overall averages
    # precision_values, recall_values, f1score_values, avg_precision, avg_recall, avg_f1score = count_PRF(epoch, valid_data=TEST_DIBCO, setting=experiment, flipped=FLIPPED, thresh=THRESHOLD)
    
    # print('Individual Precision values: ', precision_values)
    # print('Individual Recall values: ', recall_values)
    # print('Individual F1-Score values: ', f1score_values)
    
    # print(f'Overall Average Precision: {avg_precision:.4f}')
    # print(f'Overall Average Recall: {avg_recall:.4f}')
    # print(f'Overall Average F1-Score: {avg_f1score:.4f}')




# import cv2

# def visualize(epoch):
#     losses = 0
#     all_preds = []
#     all_targets = []
    
#     # For manual calculation
#     manual_precisions = []
#     manual_recalls = []
#     manual_f1_scores = []
#     manual_psnrs = []
#     opencv_psnrs = []
    
#     for i, (test_index, test_in, test_out) in enumerate(test_loader):
#         bs = len(test_in)

#         inputs = test_in.to(device)
#         outputs = test_out.to(device)

#         with torch.no_grad():
#             loss, pred_pixel_values = model(inputs, outputs)
            
#             rec_patches = pred_pixel_values
#             rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
#                                    p1=patch_size, p2=patch_size, h=image_size[0]//patch_size)
            
#             for j in range(bs):
#                 imvisualize(inputs[j].cpu(), outputs[j].cpu(), rec_images[j].cpu(), test_index[j], epoch, experiment)
                
#             losses += loss.item()
            
#             preds = (rec_images > THRESHOLD).float()
#             all_preds.extend(preds.cpu().numpy().flatten())
#             all_targets.extend(outputs.cpu().numpy().flatten())
            
#             # Manual calculation for each image in the batch
#             for j in range(bs):
#                 binary_preds = (rec_images[j].cpu().numpy() > THRESHOLD).astype(int)
#                 binary_targets = outputs[j].cpu().numpy().astype(int)
                
#                 flat_preds = binary_preds.flatten()
#                 flat_targets = binary_targets.flatten()
                
#                 TP = np.sum((flat_preds == 1) & (flat_targets == 1))
#                 FP = np.sum((flat_preds == 1) & (flat_targets == 0))
#                 FN = np.sum((flat_preds == 0) & (flat_targets == 1))
                
#                 precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#                 recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#                 f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
#                 manual_precisions.append(precision)
#                 manual_recalls.append(recall)
#                 manual_f1_scores.append(f1)
                
#                 # Manual PSNR calculation
#                 mse = np.mean((binary_targets - binary_preds) ** 2)
#                 pixel_max = 1.0
#                 psnr = 20 * np.log10(pixel_max / np.sqrt(mse)) if mse != 0 else float('inf')
#                 manual_psnrs.append(psnr)
                
#                 # OpenCV PSNR calculation
#                 psnr_opencv = cv2.PSNR(binary_targets, binary_preds)
#                 opencv_psnrs.append(psnr_opencv)
    
#     # Average manual metrics over all samples
#     avg_precision = np.mean(manual_precisions)
#     avg_recall = np.mean(manual_recalls)
#     avg_f1 = np.mean(manual_f1_scores)
#     avg_manual_psnr = np.mean(manual_psnrs)
#     avg_opencv_psnr = np.mean(opencv_psnrs)
    
#     # Scikit-learn metrics calculation
#     precision = precision_score(all_targets, all_preds, average='binary')
#     recall = recall_score(all_targets, all_preds, average='binary')
#     f1 = f1_score(all_targets, all_preds, average='binary')
    
#     print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
#     print(f'Manual Precision: {avg_precision:.4f}, Manual Recall: {avg_recall:.4f}, Manual F1-Score: {avg_f1:.4f}')
#     print(f'Manual PSNR: {avg_manual_psnr:.4f}, OpenCV PSNR: {avg_opencv_psnr:.4f}')
#     print('Valid loss: ', losses / len(test_loader))
    
# def valid_model(epoch):
#     psnr = count_psnr(epoch, valid_data=TEST_DIBCO, setting=experiment, flipped=FLIPPED, thresh=THRESHOLD)
    
#     # Calculate manual PSNR using the first batch in the validation dataset (as an example)
#     first_batch = next(iter(test_loader))
#     test_in, test_out = first_batch[1].to(device), first_batch[2].to(device)
    
#     with torch.no_grad():
#         _, pred_pixel_values = model(test_in, test_out)
#         rec_patches = pred_pixel_values
#         rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
#                                p1=patch_size, p2=patch_size, h=image_size[0]//patch_size)
    
#     # Compute manual PSNR and OpenCV PSNR for the first image
#     binary_preds = (rec_images[0].cpu().numpy() > THRESHOLD).astype(int)
#     binary_targets = test_out[0].cpu().numpy().astype(int)
    
#     mse = np.mean((binary_targets - binary_preds) ** 2)
#     manual_psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse != 0 else float('inf')
#     opencv_psnr = cv2.PSNR(binary_targets, binary_preds)
    
#     print(f'Test PSNR (from count_psnr): {psnr}')
#     print(f'Manual PSNR: {manual_psnr:.4f}, OpenCV PSNR: {opencv_psnr:.4f}')

model_name = cfg.model_weights_path
if __name__ == '__main__':
    model.load_state_dict(
        torch.load(
            # '/csehome/m23csa015/T2T-BinFormer/weights/best-model_16_2016base_256_16.pt',
            model_name,
            map_location=device))
    a = 457
    #count_parameters(model)
    epoch = "_testing"

    visualize(str(epoch))
    # utils.create_full_img(2018)
    valid_model(str(epoch))
