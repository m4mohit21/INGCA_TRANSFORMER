#!/bin/bash
# Job name:
#SBATCH --job-name=trial
# Partition:
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1

## Processors per task:
#SBATCH --cpus-per-task=4
#
#SBATCH --gres=gpu:1 ##Define number of GPUs

#
## Command(s) to run (example):
module load conda/conda 
source activate
conda activate mtp

# mpirun python process_dibco.py --data_path /scratch/m23csa015/DIBCOSETS --split_size 256 --testing_dataset 2018 --validation_dataset 2016
# # wait
python process_dibco.py --data_path /scratch/m23csa015/DIBCOSETS_INGCA/DIBCOSETS_ingca/ --data_root /scratch/m23csa015/DIBCOSETS_INGCA/  --split_size 256 --testing_dataset ingca_testing --validation_dataset 2016
wait
python train.py --data_path /scratch/m23csa015/DIBCOSETS_INGCA/DIBCOSETS_ingca/ --data_root /scratch/m23csa015/DIBCOSETS_INGCA/ --batch_size 32 --vit_model_size base --vit_patch_size 16 --epochs 1 --split_size 256 --validation_dataset 2016
# wait
# python test.py --data_path /scratch/m23csa015/DIBCOSETS_ingca/ --data_root /scratch/m23csa015/ --model_weights_path  /csehome/m23csa015/T2T-BinFormer/dibco_ingca_model/best-model_16_2016base_256_16.pt  --batch_size 8 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset ingca_testing
# wait
# python process_dibco.py --data_path /scratch/m23csa015/DIBCOSETS/ --data_root /scratch/m23csa015/ --split_size 256 --testing_dataset 2018 --validation_dataset 2016

# python test.py --data_path /scratch/m23csa015/DIBCOSETS/ --model_weights_path  /csehome/m23csa015/T2T-BinFormer/9_IMGES_MODEL/best-model_16_2016base_256_16.pt  --batch_size 8 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset 2018
# echo "starting"
python inference.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path  /csehome/m23csa015/T2T-BinFormer/9_ingca_model/model_116_2016base_256_16.pt --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
wait
# #---------------
python visualize.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path  /csehome/m23csa015/T2T-BinFormer/weights_208imgs/best-model_16_2016base_256_16.pt --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
wait
python full_img.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path  /csehome/m23csa015/T2T-BinFormer/weights_208imgs/best-model_16_2016base_256_16.pt --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
wait
# # -------------------------------------
# python visualize.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path  /csehome/m23csa015/T2T-BinFormer/dibco_ingca_model_6train/best-model_16_2016base_256_16.pt --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
# wait
# python full_img.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path  /csehome/m23csa015/T2T-BinFormer/dibco_ingca_model_6train/best-model_16_2016base_256_16.pt --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
# wait
# #---------------
# python visualize.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path  /csehome/m23csa015/T2T-BinFormer/6_train_IMGES_MODEL_only/best-model_16_2016base_256_16.pt --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
# wait
# python full_img.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path  /csehome/m23csa015/T2T-BinFormer/6_train_IMGES_MODEL_only/best-model_16_2016base_256_16.pt --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
# wait
# #---------------
# python visualize.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path  /csehome/m23csa015/T2T-BinFormer/Original_dibco_model_1/best-model_16_2016base_256_16.pt --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
# wait
# python full_img.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path  /csehome/m23csa015/T2T-BinFormer/Original_dibco_model_1/best-model_16_2016base_256_16.pt --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
wait
# -------------------------------------
# python rename.py
# python tif_to_png.py