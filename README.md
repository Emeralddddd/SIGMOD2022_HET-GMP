# HET-GMP: a Graph-based System Approach to Scaling Large Embedding Model Training (SIGMOD 2022)

## Installation
1. Clone this respository.

2. prepare build requirements:

```shell
# make sure cuda (>=10.1) is already installed in /usr/local/cuda
# create a new conda environment
conda install -c conda-forge \
cmake=3.18 zeromq=4.3.2 pybind11=2.6.0 thrust=1.11 cub=1.11 nccl=2.9.9.1 cudnn=7.6.5 openmpi=4.0.3
```

3. build

```shell
mkdir build && cd build && cmake .. && make -j && cd ..
source hetu.exp # this edits PYTHONPATH
```

4. Some python packages and necessary to run the datasets processing and training script below.

```shell
pip install --upgrade-strategy only-if-needed \
scipy sklearn numpy pyyaml argparse pandas tqdm
```

### Tips: 
If you meet the following error during the compilation:
```shell
CMake Error in hetuCTR/csrc/CMakeLists.txt:
 CUDA_ARCHITECTURES is empty for target "hetuCTR".
```
That is because you are using a different GPU and the CUDA_ARCHITECTURES needs to be changed based on the GPU type, e.g., 70 for V100, 80 for A100, and more are in this [link](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/). To solve it, please add the command (e.g., set_property(TARGET hetuCTR PROPERTY CUDA_ARCHITECTURES 70) for V100) for both /src/CMakeLists.txt and /hetuCTR/csrc/CMakeLists.txt.

## Download and process datasets

We have provided a preprocessed criteo dataset in [google drive](https://drive.google.com/file/d/1eZj8ZU_I4-6HmhgJ2_SPlJffu1Xp3T1N/view?usp=sharing).
After you download it, just execute "tar xzvf hetuctr_dataset.tar.gz" and you will find two folders including "criteo" and "partition" in the "hetuctr_dataset" folder.
The files in the first "criteo" folder are generated by the data preprocessing script "load_data.py".
And the file in the second "partition" folder is generated by the graph partition script "partition.py".
And you can also execute the partition script by yourself, e.g., "python3 partition.py -n 8 -o criteo_partition_8.npz --rerun 5". 
This is only to prevent some unknown data format errors during downloading and processing. You can directly use it for the evaluation or generate it by yourself with the following instructions:

Download criteo datasets from https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310

```shell
# in repo root directory
mkdir -p ~/hetuctr_dataset/criteo
# put your downloaded dac.tar.gz in ~/hetuctr_dataset/criteo

# copy data process script
cp ./examples/models/load_data.py ~/hetuctr_dataset/criteo
# copy graph partition script
cp ./hetuCTR/experimental/partition.py ~/hetuctr_dataset/criteo

# --------------------------------------------------------

cd ~/hetuctr_dataset/criteo

# process criteo data
python3 load_data.py

# run graph partition
# Note : you can skip this step if you only use one gpu or want to use random partition
python3 partition.py -n 8 -o criteo_partition_8.npz --rerun 5
```

Finally, you can find 6 npy file which are processed train data and a npz file which is the partition reuslt.

## Train models

Run this script to train on a single GPU:

```shell
python3 examples/hetuctr.py \
--dataset criteo --model wdl \
--batch_size 8192 --iter 1000000 --embed_dim 16 \
--val --eval_every 10000
```

Train on 8 GPUs with partition and staleness

```shell
# in repo root directory
mpirun --allow-run-as-root -np 8 \
python3 examples/hetuctr.py \
--dataset criteo --model wdl \
--batch_size 8192 --iter 1000000 --embed_dim 128 \
--partition ~/hetuctr_dataset/partition/criteo_partition_8.npz \
--store_rate 0.01 --bound 100 \
--val --eval_every 10000
```

Arguments :

​	--embed_dim : the dimension for each embedding index

​    --partition : assign a partition file, if no partition is provided, random partition is used

​    --store_rate : the amount of mirror embeddings , 0.01 means selects top 1% priority embedding as mirror embeddings on each worker

​    --bound : the staleness bound, set to 0 for BSP training, use values 10, 100 for better performance.

​    --val, --eval_every : whether to perform evaluation

​    --iter : how many iterations to run

​    --batch_size : batch size on each worker

​    --model : wdl for WideDeep model, dcn for Deep&Cross model

