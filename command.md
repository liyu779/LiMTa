This repository contains the official implementation of the paper:  
**"[Your Paper Title]"**  

## Features
- Implementation of FrqRec and MT-adapter
- Pre-training, linear evaluation, and adapter fine-tuning pipelines
- Reproducible configuration files for all experiments
- Support for  dataset

## Dataset Preparation
1. Download the dataset from the [repository](https://github.com/wangtz19/NetMamba)
2. Organize the dataset with the following structure:/dataset/CICIoT2022/{train}or{test}
## Getting Started
pip install -r requirements.txt
### Pretrain of FreqRec
bash
python main_pretrain_traffic.py --config-path scripts/pretrain/traffic/ --config-name reconstruct.yaml

Pre-trained checkpoints are available at:  [[checkpoints](https://drive.google.com/drive/folders/1u6xPO0gU3699blcttSe08zFuluC_321H?usp=sharing)]
Take it in /trained_models/
### Linear Finetune
bash
python main_linear_traffic.py --config-path scripts/linear/traffic/ --config-name reconstruct.yaml

### adapter Finetune of MT-adapter
bash
python main_adapter_traffic.py --config-path scripts/adapter/traffic/ --config-name reconstruct.yaml

## Acknowledgements
- This implementation is based on [[solo-learn repository link](https://github.com/vturrisi/solo-learn)]
- Dataset from [NetMamba repository link](https://github.com/wangtz19/NetMamba)]