
# Setup Steps by Xiaoyan

### Step 1: Git clone this github repo [SurfHead(https://github.com/Oliver-Cong02/SurFhead-Hand/tree/main)]("https://github.com/Oliver-Cong02/SurFhead-Hand/tree/main") and follow [SurfHead]("https://github.com/SurFhead2025/SurFhead") and [GaussianAvatars]("https://github.com/ShenhanQian/GaussianAvatars") to setup the conda env.

```shell
git clone https://github.com/Oliver-Cong02/SurFhead-Hand.git --recursive
cd SurFhead-Hand

conda create --name surfhand -y python=3.10
conda activate surfhand

# Install CUDA and ninja for compilation
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja  # use the right CUDA version

# Install PyTorch (make sure that the CUDA version matches with "Step 1")
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
# or
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
# make sure torch.cuda.is_available() returns True

# Install the rest packages (can take a while to compile diff-gaussian-rasterization, simple-knn, and nvdiffrast)
pip install -r requirements.txt

mkdir submodules 
cd submodules
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git --recursive
git clone https://github.com/coreqode/manopth --recursive

cd simple-knn
python -m setup install
cd ..

cd manopth
python -m setup install
cd ..

# Install other missing packages simply by pip install xxx after running the code.
pip install xxx
```

### Follow Chandradeep code [https://github.com/coreqode/manipulation/tree/main](https://github.com/coreqode/manipulation/tree/main) to run the video_seg.py to get .cache dir.

### Post-process to get more accurate masks for hand and object. (To be written...)

### Train


```shell
# please see the hyper-parameters in train.sh 
sh train.sh 0
```

# ----------------------------------------------------------------------------------------------------------------------------------------------------------





# SurFhead: Affine Rig Blending for Geometrically Accurate 2D Gaussian Surfel Head Avatars


### Installation

We heavily followed [GaussianAvatars' instruction]("https://github.com/ShenhanQian/GaussianAvatars")

Our default installation method is based on Conda package and environment management:

#### Step 1: Clone this repo and install `cuda-toolkit` with `conda`

```shell
git clone xxxx --recursive
d SurFhead-Hand

conda create --name surfhead -y python=3.10
conda activate surfhead

# Install CUDA and ninja for compilation
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja  # use the right CUDA version

# Install the rest packages (can take a while to compile diff-gaussian-rasterization, simple-knn, and nvdiffrast)
pip install -r requirements.txt
```

#### Step 2: Setup paths (for Linux)

```shell
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"
```


#### Step 3: Install PyTorch and other packages

```shell
# Install PyTorch (make sure that the CUDA version matches with "Step 1")
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
# or
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
# make sure torch.cuda.is_available() returns True

# Install the rest packages (can take a while to compile diff-gaussian-rasterization, simple-knn, and nvdiffrast)
pip install -r requirements.txt
```

## Running
In the each shell, we curated all ablation studies. Last paragraph is our final version, SurFhead.
### Training

To run the optimizer, simply use

```shell
sh train_cluster_external.sh
```

### Test (self-reenactment)

To run the optimizer, simply use

```shell
sh test_cluster_external.sh
```


<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --eval
  Add this flag to use a training/val/test split for evaluation.
  #### --bind_to_mesh
  Add this flag to bind 3D Gaussians to a driving mesh, e.g., FLAME.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```60000``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interal
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>

By default, the trained models use all available images in the dataset. To train them while withholding a validation set and a test set for evaluation, use the ```--eval``` flag. 

A complete evaluation on the validation set (novel-view synthesis) and test set (self-reenactment) will be conducted every `--interval` iterations. You can check the metrics in the terminal or within Tensorboard. Although we only save a few images in Tensorboard, the metrics are computed on all images.




