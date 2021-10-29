# Skyformer

This repository is the official implementation of [Skyformer: Remodel Self-Attention with Gaussian Kernel and Nystr\"om Method](/doc/Skyformer_camera_ready.pdf) (NeurIPS 2021). 

![image](/doc/Skyformer_model.jpg)


## Requirements

To install requirements in a conda environment:
```
conda create -n skyformer python=3.6
conda activate skyformer
pip install -r requirements.txt
```

Note: Specific requirements for data preprocessing are not included here.


# Data Preparation

Processed files can be downloaded [here](https://drive.google.com/drive/folders/1rE0SjpeFKPFtgmWWjYCoIMz91UozHWWC?usp=sharing), or processed with the following steps:

1. Requirements
```
tensorboard>=2.3.0
tensorflow>=2.3.1
tensorflow-datasets>=4.0.1
```
2. Download [the TFDS files for pathfinder](https://storage.cloud.google.com/long-range-arena/pathfinder_tfds.gz) and then set _PATHFINER_TFDS_PATH to the unzipped directory (following https://github.com/google-research/long-range-arena/issues/11)
3. Download [lra_release.gz](https://storage.googleapis.com/long-range-arena/lra_release.gz) (7.7 GB).
4. Unzip `lra-release` and put under `./data/`.
```
cd data
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar zxvf lra-release.gz 
```
5. Create a directory `lra_processed` under `./data/`.
```
mkdir lra_processed
cd ..
```
6.The directory structure would be (assuming the root dir is `code`)
```
./data/lra-processed
./data/long-range-arena-main
./data/lra_release
```
7. Create train, dev, and test dataset pickle files for each task.
```
cd preprocess
python create_pathfinder.py
python create_listops.py
python create_retrieval.py
python create_text.py
python create_cifar10.py
```

Note: most source code comes from [LRA repo](https://github.com/google-research/long-range-arena).



# Run 

Modify the configuration in `config.py` and run
```
python main.py --mode train --attn skyformer --task lra-text
```
- mode: `train`, `eval`
- attn: `softmax`, `nystrom`, `linformer`, `reformer`, `perfromer`, `informer`, `bigbird`,  `kernelized`, `skyformer`
- task: `lra-listops`, `lra-pathfinder`, `lra-retrieval`, `lra-text`, `lra-image`


## Reference

```bibtex
@inproceedings{Skyformer,
    title={Skyformer: Remodel Self-Attention with Gaussian Kernel and Nystr\"om Method}, 
    author={Yifan Chen and Qi Zeng and Heng Ji and Yun Yang},
    booktitle={NeurIPS},
    year={2021}
}
```