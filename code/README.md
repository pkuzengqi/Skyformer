
# requirements

```
python==3.6
torch==1.8.0
transformers==4.5.0
performer-pytorch
```

If you use conda, you can create an environment from this package with the following command:
```
conda env create -f environment.yml
```

Note: Specific requirements for data preprocessing are not included here.





# Data 

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
6.The directory structure would be
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

Change the configuration in `config.py` and run
```
CUDA_VISIBLE_DEVICES=0 python3 run_lra.py --mode train --attn softmax --task lra-text
```
- mode: train, eval
- attn: softmax, nystrom, linformer, reformer, perfromer, informer, bigbiard, softmaxRBF32 (`Kernelized Attention`), skecthedRBF32128（`Skyformer`)
- task: lra-listops, lra-pathfinder, lra-retrieval, lra-text


# Check Tensorboard
Run 
```
tensorboard --logdir=./log/lra_text_softmax.tensorboard --port 8123
```

Visit with browser
```
http://localhost:8123/
```
