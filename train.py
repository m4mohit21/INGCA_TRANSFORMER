from typing import Any 
import torch
from tqdm import tqdm 
from VIT import * 
from models.model import BINMODEL 
import torchvision.transforms as transforms 
import numpy as np 
import torch.optim as optim 
from einops import rearrange 
import loadData2 as loadData 
import utils as utils 
from config import Configs 
import torch_optimizer as optimizer
import os

# show attention

from vit_pytorch.recorder import Recorder
import cv2
import matplotlib.pyplot as plt



def show_attn(images,attns):
    for c_im in range(len(images)):
        im_idx = images[c_im]
        for layer in range (5,6):

            ## summ them 
            # for s in range (1,8):
            #     attns[c_im,layer,0]  = torch.mm(attns[c_im,layer,0],attns[c_im,layer,s])
            # a=41

            for head in range (2,3):
                for i in range(1025):
                    oneatt = attns[c_im,layer,head,i]
                    oneatt = oneatt[1:]
                    oneatt = oneatt.cpu()
                    oneatt = oneatt.numpy()
                    oneatt = oneatt.reshape(32,32)
                    oneatt = cv2.resize(oneatt, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

                    ## normalize ?
                    max_v = np.max(oneatt)
                    oneatt = oneatt / max_v
                    if not os.path.exists('attns/'+im_idx):
                        os.makedirs('attns/'+im_idx)
                    # oneatt = cv2.cvtColor(oneatt,cv2.COLOR_GRAY2RGB)
                    plt.imsave('attns/'+im_idx+'/layer_'+str(layer)+'_head_'+str(head)+'_patch_'+str(i)+'.png',oneatt)
        # break


cfg = Configs().parse()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

count_psnr = utils.count_psnr
imvisualize = utils.imvisualize
load_data_func = loadData.loadData_sets


transform = transforms.Compose([transforms.RandomResizedCrop(256),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

SPLITSIZE = cfg.split_size
SETTING = cfg.vit_model_size
TPS = cfg.vit_patch_size

batch_size = cfg.batch_size

experiment = SETTING +'_'+ str(SPLITSIZE)+'_' + str(TPS)       # base 

patch_size = TPS
image_size =  (SPLITSIZE,SPLITSIZE)


MASKINGRATIO = 0.5
VIS_RESULTS = True
VALID_DIBCO = cfg.validation_dataset
DATASET_PATH= cfg.data_path
TESTING_DATASET= cfg.testing_dataset


# if SETTING == 'base':
#     ENCODERLAYERS = 6
#     ENCODERHEADS = 8
#     ENCODERDIM = 256

if SETTING == 'small':
    ENCODERLAYERS = 3
    ENCODERHEADS = 4
    ENCODERDIM = 512

if SETTING == 'large':
    ENCODERLAYERS = 12
    ENCODERHEADS = 16
    ENCODERDIM = 1024

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


def all_data_loader():
    data_train, data_valid, data_test = load_data_func()
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, valid_loader, test_loader

trainloader, validloader, testloader = all_data_loader()

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
    decoder_heads = ENCODERHEADS       # anywhere from 1 to 8
)


model = model.to(device)

optimizer = optim.AdamW(model.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)
#optimizer = optimizer.DiffGrad(
#    model.parameters(),
#    lr= 1e-3,
#    betas=(0.9, 0.999),
#    eps=1e-8,
#    weight_decay=0,
#)
#-----------------------------------------------------------
# # Save the model checkpoint for resuming
# def save_checkpoint(state, filename='checkpoint.pth.tar'):
#     print(f"=> Saving checkpoint at {filename}")
#     torch.save(state, filename)

# # Load the model checkpoint for resuming
# def load_checkpoint(checkpoint, model, optimizer):
#     print(f"=> Loading checkpoint from {checkpoint}")
#     state = torch.load(checkpoint)
#     model.load_state_dict(state['state_dict'])
#     optimizer.load_state_dict(state['optimizer'])
#     epoch = state['epoch']
#     best_psnr = state['best_psnr']
#     print(f"=> Loaded checkpoint from epoch {epoch} with best PSNR: {best_psnr}")
#     return epoch, best_psnr

#------------------------------------------------------------------

def visualize(epoch):
    losses = 0
    for i, (valid_index, valid_in, valid_out) in enumerate(validloader):
        # inputs, labels = data
        bs = len(valid_in)
        #print("BS: ",bs)

        inputs = valid_in.to(device)
        outputs = valid_out.to(device)
        
        with torch.no_grad():
            loss, pred_pixel_values = model(inputs,outputs)
            #print("Inside Visualize: ",pred_pixel_values)
            rec_patches = pred_pixel_values
            #print("Rec Patches: ",rec_patches.shape)

            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
            #print("Reconstructed Image: ",rec_images)
            
            for j in range (0,bs):
                imvisualize(inputs[j].cpu(),outputs[j].cpu(),rec_images[j].cpu(),valid_index[j],epoch,experiment)
            
            losses += loss.item()
            #v = Recorder(model.encoder)
            #preds,attns = v(inputs)
            #print(attns)
        #show_attn(valid_index, attns)
        
    
    print('valid loss: ', losses / len(validloader))



def valid_model(epoch):
    global best_psnr
    global best_epoch

    print('last best psnr: ', best_psnr, 'epoch: ', best_epoch)
    if not os.path.exists('./weights/'):
        os.makedirs('./weights/')
    torch.save(model.state_dict(), f'./weights/model_{epoch}_'+str(TPS)+'_'+VALID_DIBCO+experiment+'.pt')

    psnr  = count_psnr(DATASET_PATH,VALID_DIBCO,epoch,valid_data=VALID_DIBCO,setting=experiment)
    print('curr psnr: ', psnr)


    if psnr >= best_psnr:
        best_psnr = psnr
        best_epoch = epoch
        

    
        torch.save(model.state_dict(), './weights/best-model_'+str(TPS)+'_'+VALID_DIBCO+experiment+'.pt')

        dellist = os.listdir('vis'+experiment)
        dellist.remove('epoch'+str(epoch))

        for dl in dellist:
            os.system('rm -r vis'+experiment+'/'+dl)
    else:
        os.system('rm -r vis'+experiment+'/epoch'+str(epoch))
    

#------------------------------------------------------- 
# if __name__ == '__main__':

#     # Check if a checkpoint exists to resume training
#     checkpoint_path = './weights/checkpoint.pth.tar'
#     start_epoch = 1  # Start from the first epoch unless a checkpoint exists
#     best_psnr = 0  # Initialize the best PSNR value

#     if os.path.exists(checkpoint_path):
#         start_epoch, best_psnr = load_checkpoint(checkpoint_path, model, optimizer)
#     else:
#         best_psnr = 0
   

#     # Skipping the full training loop for faster testing:
#     # Just simulate one epoch or partial training for testing checkpoint saving/resuming

#     # Uncomment this to simulate a single epoch for faster testing
#     epoch = start_epoch  # Start from where the checkpoint left off

#     print(f"[TESTING EPOCH]: {epoch}")

#     # Simulate a single training step (or very small portion of it)
#     running_loss = 0.0
#     for i in range(1):  # This can be adjusted to simulate fewer iterations for faster testing
#         inputs = torch.randn(1, 3, 256, 256).to(device)  # Dummy input
#         outputs = torch.randn(1, 3, 256, 256).to(device)  # Dummy output
#         optimizer.zero_grad()
#         loss = torch.mean((inputs - outputs) ** 2)  # Simulate a simple loss calculation
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print('[Epoch: %d, Iter: %5d] Simulated Train loss: %.3f' % (epoch, 1, running_loss))

#     # Simulate visualization and validation (skip if not needed for testing)
#     # visualize(str(epoch))
#     # valid_model(epoch)

#     # Save a checkpoint after one epoch (for testing)
#     checkpoint_state = {
#         'epoch': epoch + 1,  # Update to the next epoch
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'best_psnr': best_psnr,  # This can be any metric, e.g., PSNR or validation loss
#     }
#     save_checkpoint(checkpoint_state, filename=checkpoint_path)

#     print(f"Checkpoint for epoch {epoch} saved.")

#     # Now simulate resuming from the saved checkpoint to test the loading process
#     start_epoch, best_psnr = load_checkpoint(checkpoint_path, model, optimizer)

#     print(f"Resumed training from epoch {start_epoch}, with best PSNR: {best_psnr}")
#     # # Start the training loop, accounting for possible resuming
#     # for epoch in range(start_epoch, cfg.epochs):
#     #     print(f"[EPOCH]: {epoch}")

#     #     running_loss = 0.0

#     #     for i, (train_index, train_in, train_out) in tqdm(enumerate(trainloader), total=len(trainloader)):

#     #         inputs = train_in.to(device)
#     #         outputs = train_out.to(device)

#     #         optimizer.zero_grad()
#     #         loss, _ = model(inputs, outputs)

#     #         loss.backward()
#     #         optimizer.step()

#     #         running_loss += loss.item()

#     #         show_every = int(len(trainloader) / 7)

#     #         if i % show_every == show_every-1:
#     #             print('[Epoch: %d, Iter: %5d] Train loss: %.3f' % (epoch, i + 1, running_loss / show_every))
#     #             running_loss = 0.0

#     #     # After each epoch, visualize and validate the model
#     #     if VIS_RESULTS:
#     #         visualize(str(epoch))
#     #         valid_model(epoch)

#     #     # Save a checkpoint after each epoch
#     #     checkpoint_state = {
#     #         'epoch': epoch + 1,
#     #         'state_dict': model.state_dict(),
#     #         'optimizer': optimizer.state_dict(),
#     #         'best_psnr': best_psnr,
#     #     }
#     #     save_checkpoint(checkpoint_state, filename=checkpoint_path)
#-------------------------------------------------------
if __name__ == '__main__':
    for epoch in range(1,cfg.epochs):
        print(f"[EPOCH]: epoch")

        running_loss = 0.0

        for i, (train_index, train_in, train_out) in tqdm(enumerate(trainloader),total = len(trainloader)):

            inputs = train_in.to(device)
            outputs = train_out.to(device)
            #print("Input Shape: ",inputs.shape)
            optimizer.zero_grad()
            loss,_ = model(inputs,outputs)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            show_every = int(len(trainloader) / 7)
            #print(running_loss)

            if i % show_every == show_every-1:    # print every 20 mini-batches
                print('[Epoch: %d, Iter: %5d] Train loss: %.3f' % (epoch, i + 1, running_loss / show_every))
                running_loss = 0.0
        
        
        if VIS_RESULTS:
            visualize(str(epoch))
            valid_model(epoch)


