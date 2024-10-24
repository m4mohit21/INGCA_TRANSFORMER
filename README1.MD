Here's a more structured and comprehensive README file based on your project directory structure and the `.gitignore` file, similar to the example you provided:

---

# 🥇 **Document Image Binarization with T2T-BinFormer**

This repository contains the implementation of the modified **Tokens-to-Token Transformer (T2T-BinFormer)** model for document image binarization tasks. This version introduces several improvements aimed at better processing, faster inference, and scalability for datasets like DIBCO and custom datasets.

## 📜 **Project Overview**

The **T2T-BinFormer** model is designed to perform document image binarization using a Transformer-based approach. The model splits input images into patches and processes them through an encoder-decoder architecture. The key focus of this repository is on improving the model’s performance on document image binarization, optimizing patch-based processing, and refining the inference pipeline to handle large datasets efficiently.

## 🛠️ **Requirements**

To replicate the environment used for this project, you can either create a conda environment or manually install the dependencies. Follow the steps below to set up your environment.

### ⚙️ **Conda Environment Setup**

To set up a new environment for this project using Conda:

```bash
# Create a new environment with Python 3.8
conda create --name dibco_image_binarization python=3.8

# Activate the environment
conda activate dibco_image_binarization

# Install the dependencies from requirements.txt
pip install -r requirements.txt
```

### 🔑 **Key Libraries**

- **Python**: 3.8
- **PyTorch**: 1.12.0
- **NumPy**: 1.21.2
- **Matplotlib**: 3.4.3
- **OpenCV**: 4.5.3
- **Pillow (PIL)**: 8.3.2
- **tqdm**: 4.62.3

Other dependencies can be found in the `requirements.txt` file.

---

## 📁 **Project Structure**

```plaintext
.
├── data/                        # Data directory containing training and testing images
├── models/                      # Directory for model checkpoints and saved weights
├── scripts/                     # Python scripts for data preprocessing, training, inference, etc.
│   ├── process_dibco.py         # Script for preprocessing images (splitting into patches)
│   ├── train.py                 # Script for training the model
│   ├── inference.py             # Script for inference on test images
│   ├── visualize.py             # Script for visualizing binarized image patches
│   └── full_img.py              # Script for reconstructing full images from patches
├── results/                     # Directory to store output and visualization results
├── mtp.sh                       # SLURM job script for running the full pipeline
├── requirements.txt             # Required libraries and dependencies
└── README1.md                    # Project description and setup instructions
```

---

## 🚀 **Running the Project**

### Step 1: **Preprocessing**

Before training, you need to split the dataset into patches for training and validation. Use the following command to preprocess your DIBCO dataset:

```bash
python process_dibco.py --data_path /scratch/m23csa015/DIBCOSETS_INGCA/DIBCOSETS_ingca/ --data_root /scratch/m23csa015/DIBCOSETS_INGCA/ --split_size 256 --testing_dataset ingca_testing --validation_dataset 2016
```

### Step 2: **Training**

To train the model using preprocessed patches, run the `train.py` script. Model weights will be saved after training:

```bash
python train.py --data_path /scratch/m23csa015/DIBCOSETS_INGCA/DIBCOSETS_ingca/ --batch_size 32 --vit_model_size base --vit_patch_size 16 --epochs 1 --split_size 256 --validation_dataset 2016
```

### Step 3: **Inference**

For predicting the binarization results on test images, use the `inference.py` script to split test images into patches and predict each patch's result:

```bash
python inference.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path /csehome/m23csa015/T2T-BinFormer/9_ingca_model/model_116_2016base_256_16.pt --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
```

### Step 4: **Visualization**

To visualize the predicted binarized patches, run the `visualize.py` script:

```bash
python visualize.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path /csehome/m23csa015/T2T-BinFormer/weights_208imgs/best-model_16_2016base_256_16.pt --batch_size 16 --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
```

### Step 5: **Reconstruction**

Once the patches are processed, you can reconstruct them into full-sized images using the `full_img.py` script:

```bash
python full_img.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --data_root /scratch/m23csa015/INGCA/ --model_weights_path /csehome/m23csa015/T2T-BinFormer/weights_208imgs/best-model_16_2016base_256_16.pt --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
```

---

## 🧾 **SLURM Job Script**

The entire process can also be automated using a SLURM job script. Below is an example of the `mtp.sh` script for running preprocessing, training, and inference on a SLURM-based cluster:

```bash
#!/bin/bash
#SBATCH --job-name=trial
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module load conda/conda
source activate
conda activate mtp

# Preprocess dataset into patches
python process_dibco.py --data_path /scratch/m23csa015/DIBCOSETS_INGCA/DIBCOSETS_ingca/ --data_root /scratch/m23csa015/DIBCOSETS_INGCA/ --split_size 256 --testing_dataset ingca_testing --validation_dataset 2016
wait

# Train model
python train.py --data_path /scratch/m23csa015/DIBCOSETS_INGCA/DIBCOSETS_ingca/ --batch_size 32 --vit_patch_size 16 --epochs 1 --split_size 256 --validation_dataset 2016
wait

# Inference
python inference.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --model_weights_path /csehome/m23csa015/T2T-BinFormer/9_ingca_model/model_116_2016base_256_16.pt --batch_size 16 --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
wait

# Visualize binarized patches
python visualize.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --model_weights_path /csehome/m23csa015/T2T-BinFormer/weights_208imgs/best-model_16_2016base_256_16.pt --batch_size 16 --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
wait

# Reconstruct full images
python full_img.py --data_path /scratch/m23csa015/INGCA/INGCA_IMAGES/ --model_weights_path /csehome/m23csa015/T2T-BinFormer/weights_208imgs/best-model_16_2016base_256_16.pt --batch_size 16 --vit_patch_size 16 --split_size 256 --testing_dataset raw_images
```

---

## 📜 **.gitignore**

The `.gitignore` file excludes the following files and folders from version control:

```plaintext
# Ignore datasets and large files
data/
mY_OWN_DATASET/
diBCO_OVER_MY_DATASET/
gt/

# Ignore specific files and directories
rename.py
weights_old/
visbase_256_16/
cleaned_mtp.sh

# Ignore output and log files
output.txt
out.txt
*.out
slurm-*.out

# Ignore cache directories
__pycache__/

# Ignore text files with trial data
trail_1.txt

# Ignore directories starting with 'visinference_'
visinference_*/
```

---

