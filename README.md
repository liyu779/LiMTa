# LiMTA 
This repository contains the implementation for the paper **"Towards Efficient Distributed Network Security: A Lightweight Multitask Traffic Analysis Framework"**.

Our code is based on the [solo-learn repository](https://github.com/vturrisi/solo-learn).


### ✨ Features

- **Implementation of FrqRec and MT-adapter** (Frequency Reconstruction and Multitask Adapter).
    
- Comprehensive pipelines for:
    
    - **Pre-training**
        
    - **Linear Evaluation**
        
    - **Adapter Fine-tuning**
        
- Reproducible configuration files for all experiments.
    

### 💾 Dataset Preparation

1. Download the **dataset** from the [NetMamba repository](https://github.com/wangtz19/NetMamba).
    
2. Organize the dataset with the following directory structure:
    
    ```
    /dataset/CICIoT2022/
    ├── train/
    └── test/
    ```
    

### 🚀 Getting Started

First, install the required dependencies:

```
pip install -r requirements.txt
```

#### 1. Pre-training with FreqRec

Run the pre-training script using the reconstruction configuration:

```
python main_pretrain_traffic.py --config-path scripts/pretrain/traffic/ --config-name reconstruct.yaml
```

> **Note:** **Pre-trained checkpoints** are available for download [here](https://www.google.com/search?q=https://drive.google.com/drive/folders/1u6xPO0gU3699blcttSe08fFuluC_321H%3Fusp%3Dsharing). Place the downloaded checkpoints in the `/trained_models/` directory.

#### 2. Linear Fine-tuning

Use the pre-trained model for linear evaluation (fine-tuning the classification head only):

```
python main_linear_traffic.py --config-path scripts/linear/traffic/ --config-name reconstruct.yaml
```

#### 3. MT-adapter Fine-tuning

Fine-tune the model using the MT-adapter:

```
python main_adapter_traffic.py --config-path scripts/adapter/traffic/ --config-name reconstruct.yaml
```


### 🙏 Acknowledgements

- **solo-learn**: Our implementation builds upon the great work from the [solo-learn repository](https://github.com/vturrisi/solo-learn).
    
- **NetMamba**: We use the dataset provided by the [NetMamba repository](https://github.com/wangtz19/NetMamba).
