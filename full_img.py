#pipline steps
import time
import numpy as np
import os
from PIL import Image
import random
import cv2
from shutil import copy
from tqdm import tqdm
from config import Configs
import shutil


import torch.utils.data as D
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import os
from PIL import Image
from config import Configs


import matplotlib.pyplot as plt
import os
import numpy as np
import math
from tqdm import tqdm
import cv2
from config import Configs
from sklearn.metrics import precision_score, recall_score, f1_score


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
# import loadData2 as loadData
# import utils as utils
from  config import Configs
import os
from sklearn.metrics import precision_score, recall_score, f1_score


cfg = Configs().parse()

def prepare_infrence(folder, test_set, patches_size, overlap_size,
                             patches_size_valid):
 

    all_datasets = os.listdir(folder)
    print(all_datasets)
    n_i = 1

    for d_set in tqdm(all_datasets):
        
        if d_set == test_set:
            for im in os.listdir(folder + d_set + '/imgs'):
                print(im)
                img = cv2.imread(folder + d_set + '/imgs/' + im)
                # gt_img = cv2.imread(folder + d_set + '/gt_imgs/' + im)

                for i in range(0, img.shape[0], patches_size_valid):
                    for j in range(0, img.shape[1], patches_size_valid):

                        if i + patches_size_valid <= img.shape[
                                0] and j + patches_size_valid <= img.shape[1]:
                            p = img[i:i + patches_size_valid,
                                    j:j + patches_size_valid, :]
                            # gt_p = gt_img[i:i + patches_size_valid,
                            #               j:j + patches_size_valid, :]

                        elif i + patches_size_valid > img.shape[
                                0] and j + patches_size_valid <= img.shape[1]:
                            p = np.ones((patches_size_valid,
                                         patches_size_valid, 3)) * 255
                            # gt_p = np.ones((patches_size_valid,
                            #                 patches_size_valid, 3)) * 255

                            p[0:img.shape[0] -
                              i, :, :] = img[i:img.shape[0],
                                             j:j + patches_size_valid, :]
                            # gt_p[0:img.shape[0] -
                                #  i, :, :] = gt_img[i:img.shape[0],
                                #                    j:j + patches_size_valid, :]

                        elif i + patches_size_valid <= img.shape[
                                0] and j + patches_size_valid > img.shape[1]:
                            p = np.ones((patches_size_valid,
                                         patches_size_valid, 3)) * 255
                            # gt_p = np.ones((patches_size_valid,
                            #                 patches_size_valid, 3)) * 255

                            p[:, 0:img.shape[1] -
                              j, :] = img[i:i + patches_size_valid,
                                          j:img.shape[1], :]
                            # gt_p[:, 0:img.shape[1] -
                            #      j, :] = gt_img[i:i + patches_size_valid,
                            #                     j:img.shape[1], :]

                        else:
                            p = np.ones((patches_size_valid,
                                         patches_size_valid, 3)) * 255
                            # gt_p = np.ones((patches_size_valid,
                            #                 patches_size_valid, 3)) * 255

                            p[0:img.shape[0] - i,
                              0:img.shape[1] - j, :] = img[i:img.shape[0],
                                                           j:img.shape[1], :]
                            # gt_p[0:img.shape[0] - i, 0:img.shape[1] -
                            #      j, :] = gt_img[i:img.shape[0],
                            #                     j:img.shape[1], :]

                        cv2.imwrite(
                            main_path + 'test/' + im.split('.')[0] + '_' +
                            str(i) + '_' + str(j) + '.png', p)
                        # cv2.imwrite(
                        #     main_path + 'test_gt/' + im.split('.')[0] + '_' +
                        #     str(i) + '_' + str(j) + '.png', gt_p)
                        # print(main_path + 'test_gt/' + im.split('.')[0] + '_' + str(i) + '_' + str(j) + '.png')

       


class Read_data(D.Dataset):
    def __init__(self, file_label, set, augmentation=True, flipped=False):
        self.file_label = file_label
        self.set = set
        self.augmentation = augmentation
        self.flipped = flipped

    def __getitem__(self, index):
        img_name = self.file_label[index]

        idx, deg_img = self.readImages(img_name)
        return idx, deg_img

    def __len__(self):
        return len(self.file_label)

    def readImages(self, file_name):
        file_name = file_name
        url_deg = baseDir + '/' + self.set + '/' + file_name
        # url_gt = baseDir + '/' + self.set + '_gt/' + file_name

        deg_img = cv2.imread(url_deg, 0)

        # gt_img = cv2.imread(url_deg, 0)
        # gt_img = cv2.imread(url_gt, 0)

        if self.flipped:
            deg_img = cv2.rotate(deg_img, cv2.ROTATE_180)
            # gt_img = cv2.rotate(gt_img, cv2.ROTATE_180)

        try:
            deg_img.any()
        except:
            print('###!Cannot find image: ' + url_deg)

        # try:
        #     gt_img.any()
        # except:
        #     print('###!Cannot find image: ' + url_gt)

        deg_img = Image.fromarray(np.uint8(deg_img))
        # gt_img = Image.fromarray(np.uint8(gt_img))

        if self.augmentation:  # augmentation for training data

            # # Resize -->  it will affect the performance :3
            # resize = transforms.Resize(size=(256+64, 256+64))

            # deg_img = resize(deg_img)
            # gt_img = resize(gt_img)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                deg_img, output_size=(split_size, split_size))

            deg_img = TF.crop(deg_img, i, j, h, w)
            # gt_img = TF.crop(gt_img, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                deg_img = TF.hflip(deg_img)
                # gt_img = TF.hflip(gt_img)

            # Random vertical flipping
            if random.random() > 0.5:
                deg_img = TF.vflip(deg_img)
                # gt_img = TF.vflip(gt_img)

        deg_img = (np.array(deg_img) / 255).astype('float32')
        # gt_img = (np.array(gt_img) / 255).astype('float32')

        # if VGG_NORMAL:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        deg_img = deg_img[..., np.newaxis]
        # gt_img = gt_img[..., np.newaxis]
        out_deg_img = np.zeros([1, *deg_img.shape[:-1]])
        # out_gt_img = np.zeros([1, *gt_img.shape[:-1]])

        for i in range(1):
            out_deg_img[i] = (deg_img[:, :, i] - mean[i]) / std[i]
            # out_gt_img[i] = (gt_img[:, :, i] - mean[i]) / std[i]

        return file_name, out_deg_img


def loadData_sets(flipped=False):

   
    data_te = os.listdir(baseDir + '/test')
    np.random.shuffle(data_te)

   
    data_test = Read_data(data_te, 'test', augmentation=False)

    return data_test



# def imvisualize(imdeg, imgt='None', impred, ind, epoch='0', setting=''):
def imvisualize(imdeg, impred, ind, imgt='None', epoch='0', setting=''):


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    imdeg = imdeg.numpy()
    # imgt = imgt.numpy()
    impred = impred.numpy()

    imdeg = np.transpose(imdeg, (1, 2, 0))
    # imgt = np.transpose(imgt, (1, 2, 0))
    impred = np.transpose(impred, (1, 2, 0))

    for ch in range(1):
        imdeg[:, :, ch] = (imdeg[:, :, ch] * std[ch]) + mean[ch]
        # imgt[:, :, ch] = (imgt[:, :, ch] * std[ch]) + mean[ch]
        impred[:, :, ch] = (impred[:, :, ch] * std[ch]) + mean[ch]

    impred[np.where(impred > 1)] = 1
    impred[np.where(impred < 0)] = 0

    if not os.path.exists('./vis' + setting + '/epoch' + epoch):
        os.makedirs('./vis' + setting + '/epoch' + epoch)


    ### binarize

    impred = (impred > 0.5) * 1
    # cv2.imwrite(
    #     './vis' + setting + '/epoch' + epoch + '/' + str(ind).split('.')[0] +
    #     '_deg.png', imdeg * 255)

    # cv2.imwrite(
    #     './vis' + setting + '/epoch' + epoch + '/' + ind.split('.')[0] +
    #     '_deg.png', imdeg * 255)
    # cv2.imwrite(
    #     './vis' + setting + '/epoch' + epoch + '/' + ind.split('.')[0] +
    #     '_gt.png', imgt * 255)
    print('./vis' + setting + '/epoch' + epoch + '/' + str(ind).split('.')[0] +
        '_pred.png')
    cv2.imwrite(
        './vis' + setting + '/epoch' + epoch + '/' + str(ind).split('.')[0] +
        '_pred.png', impred * 255)

def create_full_img(valid_data,epoch, setting = '',flipped = False, thresh=0.5):

    flip_status = 'flipped' if flipped else 'normal'

    if not os.path.exists('vis' + setting + '/epoch' + str(epoch) +
                          '/00_reconstr_' + flip_status):
        os.makedirs('vis' + setting + '/epoch' + str(epoch) + '/00_reconstr_' +
                    flip_status)

    gt_folder = cfg.data_path + f'{valid_data}/imgs/'

    gt_imgs = os.listdir(gt_folder)
    print(gt_imgs)
    
    for im in gt_imgs:
        gt_image = cv2.imread(gt_folder + '/' + im, cv2.IMREAD_GRAYSCALE)
        max_p = np.max(gt_image)
        gt_image = gt_image / max_p

        
        pred_image = reconstruct(im.split('.')[0],
                                 gt_image.shape[0],
                                 gt_image.shape[1],
                                 epoch,
                                 setting,
                                 flipped=flipped) / max_p
        
        pred_image = (pred_image > thresh) * 1
        
        cv2.imwrite(
            'vis' + setting + '/epoch' + str(epoch) + '/00_reconstr_' +
            flip_status + '/' + im.split('.')[0] + '_pred.png',
            pred_image * 255)


def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))


def reconstruct(idx, h, w, epoch, setting, flipped=False):

    rec_image = np.zeros(((h // SPLITSIZE + 1) * SPLITSIZE,
                          (w // SPLITSIZE + 1) * SPLITSIZE, 3))

    for i in tqdm(range(0, h, SPLITSIZE)):
        for j in range(0, w, SPLITSIZE):
            # print('./vis' + setting + '/epoch' + str(epoch) + '/' +
            #                idx + '_' + str(i) + '_' + str(j) + '_pred.png')
            p = cv2.imread('./vis' + setting + '/epoch' + str(epoch) + '/' +
                           idx + '_' + str(i) + '_' + str(j) + '_pred.png')
            if flipped:
                p = cv2.rotate(p, cv2.ROTATE_180)
            rec_image[i:i + SPLITSIZE, j:j + SPLITSIZE, :] = p

    rec_image = rec_image[:h, :w, :]
    return rec_image


def count_psnr(epoch,
               valid_data='2013',
               setting='',
               flipped=False,
               thresh=0.5):
    avg_psnr = 0
    qo = 0
    print(valid_data)
    folder = 'vis/epoch' + str(epoch)
    gt_folder = 'C:/Users/Risab/Desktop/Research/Image_Binarization/DIBCOSETS/2019/gt_imgs'

    gt_imgs = os.listdir(gt_folder)

    flip_status = 'flipped' if flipped else 'normal'

    # pred_imgs = os.listdir(folder)
    # pred_imgs = [im for im  in pred_imgs if 'pred' in im]

    if not os.path.exists('vis' + setting + '/epoch' + str(epoch) +
                          '/00_reconstr_' + flip_status):
        os.makedirs('vis' + setting + '/epoch' + str(epoch) + '/00_reconstr_' +
                    flip_status)

    for im in gt_imgs:
        # if im =="image_1.png":
        #     continue

        gt_image = cv2.imread(gt_folder + '/' + im)
        max_p = np.max(gt_image)
        gt_image = gt_image / max_p

        pred_image = reconstruct(im.split('.')[0],
                                 gt_image.shape[0],
                                 gt_image.shape[1],
                                 epoch,
                                 setting,
                                 flipped=flipped) / max_p
        # pred_image = cv2.rotate(pred_image, cv2.ROTATE_180)

        pred_image = (pred_image > thresh) * 1
        avg_psnr += psnr(pred_image, gt_image)
        qo += 1

        cv2.imwrite(
            'vis' + setting + '/epoch' + str(epoch) + '/00_reconstr_' +
            flip_status + '/' + im, gt_image * 255)
        cv2.imwrite(
            'vis' + setting + '/epoch' + str(epoch) + '/00_reconstr_' +
            flip_status + '/' + im.split('.')[0] + '_pred.png',
            pred_image * 255)

    return (avg_psnr / qo)


def count_psnr_both(epoch, valid_data='2016', setting='', thresh=0.5):
    avg_psnr = 0
    qo = 0
    folder = './vis/epoch' + str(epoch)
    gt_folder = 'data/DIBCOSETS/' + valid_data + '/gt_imgs'

    gt_imgs = os.listdir(gt_folder)

    if not os.path.exists('./vis' + setting + '/epoch' + str(epoch) +
                          '/00_reconstr_merge'):
        os.makedirs('./vis' + setting + '/epoch' + str(epoch) +
                    '/00_reconstr_merge')

    for im in gt_imgs:

        gt_image = cv2.imread(gt_folder + '/' + im)
        max_p = np.max(gt_image)
        gt_image = gt_image / max_p

        pred_image1 = cv2.imread('./vis' + setting + '/epoch' + str(epoch) +
                                 '/00_reconstr_flipped/' +
                                 im.replace('.png', '_pred.png')) / max_p
        pred_image2 = cv2.imread('./vis' + setting + '/epoch' + str(epoch) +
                                 '/00_reconstr_normal/' +
                                 im.replace('.png', '_pred.png')) / max_p

        pred_image1 = (pred_image1 > thresh) * 1
        pred_image2 = (pred_image2 > thresh) * 1

        pred_image = np.ones(
            (gt_image.shape[0], gt_image.shape[1], gt_image.shape[2]))

        for i in range(gt_image.shape[0]):
            for j in range(gt_image.shape[1]):
                for k in range(gt_image.shape[2]):
                    if pred_image1[i, j, k] == 1 or pred_image2[i, j, k] == 1:
                        pred_image[i, j, k] = 1
                    else:
                        pred_image[i, j,
                                   k] = pred_image1[i, j,
                                                    k] + pred_image2[i, j, k]

        avg_psnr += psnr(pred_image, gt_image)
        qo += 1

        cv2.imwrite(
            './vis' + setting + '/epoch' + str(epoch) + '/00_reconstr_merge' +
            '/' + im, gt_image * 255)
        cv2.imwrite(
            './vis' + setting + '/epoch' + str(epoch) + '/00_reconstr_merge' +
            '/' + im.split('.')[0] + '_pred.png', pred_image * 255)

    return (avg_psnr / qo)


def sort_batch(batch):
    n_batch = len(batch)
    train_index = []
    train_in = []
    # train_out = []
    for i in range(n_batch):
        idx, img = batch[i]

        train_index.append(idx)
        train_in.append(img)
        # train_out.append(gt_img)

    train_index = np.array(train_index)
    train_in = np.array(train_in, dtype='float32')
    # train_out = np.array(train_out, dtype='float32')

    train_in = torch.from_numpy(train_in)
    # train_out = torch.from_numpy(train_out)

    return train_index, train_in

def test_data_loader():
    data_test = load_data_func(flipped=FLIPPED)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return  test_loader


def visualize(epoch, setting):
    losses = 0
    all_preds = []
    all_targets = []
    for i, (test_index, test_in) in enumerate(test_loader):
        # inputs, labels = data
        bs = len(test_in)
        print(test_index)
        # break
        inputs = test_in.to(device)

        with torch.no_grad():
            _ , pred_pixel_values = model(inputs)
            
            rec_patches = pred_pixel_values

            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
            
            for j in range (0,bs):
                imvisualize(imdeg=inputs[j].cpu(),impred =rec_images[j].cpu(),ind= test_index[j],epoch=epoch,setting = setting)
                #imdeg, impred, ind, imgt='None', epoch='0', setting=''
                # break







main_path = cfg.data_root
# validation_dataset = cfg.validation_dataset
testing_dataset = cfg.testing_dataset

# step1 first put the infrence images in the test dataset and split this images in  the patches 

if not os.path.exists(main_path + 'test/'):
    os.makedirs(main_path + 'test/')
# if not os.path.exists(main_path + 'test_gt/'):
#     os.makedirs(main_path + 'test_gt/')
'''os.system('rm ' + main_path + 'train/*')
os.system('rm ' + main_path + 'train_gt/*')

os.system('rm ' + main_path + 'valid/*')
os.system('rm ' + main_path + 'valid_gt/*')

os.system('rm ' + main_path + 'test/*')
os.system('rm ' + main_path + 'test_gt/*')'''

patch_size = cfg.split_size

p_size = (patch_size + 128)
p_size_valid = patch_size
overlap_size = patch_size // 2

folder = cfg.data_path

# ----------------------------------------------------------------------
# exit(0)

# step2 load_dataset


cfg = Configs().parse()

split_size = cfg.split_size

# baseDir = '/scratch/m23csa015/DIBCOSETS'#'--PATH TO YOUR TRAIN/TEST?VALID FOLDER--'
baseDir = cfg.data_root

#-------------------------------------------------------

cfg = Configs().parse()

SPLITSIZE = cfg.split_size




# -----------------------------------------------
# step2 after spliting the images, predicts the reconstructed images which we get in patches

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
print(device)

#
imvisualize = imvisualize
load_data_func = loadData_sets

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
# optimizer = optim.AdamW(model.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)

model_name = cfg.model_weights_path



# step3 we prepare new folder in which all the predcited images reconstructe and save



if __name__ == '__main__':
    # model.load_state_dict(
    #     torch.load(
    #         # '/csehome/m23csa015/T2T-BinFormer/weights/best-model_16_2016base_256_16.pt',
    #         model_name,
    #         map_location=device))
    a = 457
    #count_parameters(model)
    epoch = "_testing"
    SETTING=f'inference_{model_name.split("/")[-2]}'
    print(SETTING)
    # imgs_list = [f"image_{i}.png" for i in range(1001,1101)]
    # imgs_list = [f"image_{i}.jpg" for i in range(1,32)]
    # imgs_list= ['BISM-43_Scan_0150.png', 'BORI-003_Scan_0019.png', 'BORI-003_Scan_0295.png', 'BORI-003_Scan_0388.png', 'BORI-003_Scan_0627.png', 'BORI-009_Scan_0017.png', 'BORI-009_Scan_0145.png', 'BORI-009_Scan_0254.png', 'BORI-009_Scan_0545.png', 'BORI-028_Scan_0069.png', 'BORI-028_Scan_0586.png', 'BORI_003_Scan_0287.png', 'RORI_Jodhpur_001_Scan_0008.png', 'RORI_Jodhpur_001_Scan_0009.png']
    # os.listdir(os.path.join(cfg.data_path, cfg.testing_dataset, "imgs2"))
    # for img_name in imgs_list:
        # print(os.path.join(cfg.data_path, cfg.testing_dataset, "imgs2", img_name))
        # shutil.move(os.path.join(cfg.data_path, cfg.testing_dataset, "imgs2", img_name), os.path.join(cfg.data_path, cfg.testing_dataset, "imgs"))
        # print("splitting imgs")
        # prepare_infrence(folder,testing_dataset, p_size, overlap_size, p_size_valid)

        # time.sleep(10)
        # print("inferencing",flush=True)
        # visualize(str(epoch),SETTING)
        # time.sleep(5)
    print("prediction done",flush=True)
    create_full_img(testing_dataset,epoch = epoch, setting = SETTING,thresh=0.5)
    print("Full_img done")
        # shutil.move(os.path.join(cfg.data_path, cfg.testing_dataset, "imgs", img_name), os.path.join(cfg.data_path, cfg.testing_dataset, "imgs_over"))
        # print("moving done")
        # # shutil.move(os.path.join(cfg.data_path, cfg.testing_dataset, "imgs", img_name), os.path.join(cfg.data_path, cfg.testing_dataset, "imgs_over"))
        
        # for split_name in os.listdir(os.path.join(cfg.data_root, "test")):
        #     shutil.move(os.path.join(cfg.data_root, "test", split_name), os.path.join(cfg.data_root, "test_over_split"))
        # print("clean up test")
        # for split_name in os.listdir('./vis' + SETTING + '/epoch' + epoch + '/' ):
        #     if os.path.isfile(os.path.join('./vis' + SETTING + '/epoch' + epoch + '/', split_name)):
        #         os.remove(os.path.join('./vis' + SETTING + '/epoch' + epoch + '/', split_name))
        # print("clean up pred patches")
        # break
        # break