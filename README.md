# RF-3DGS: Radio Frequency 3D Gaussian Splatting

## Overview

RF-3DGS is an innovative method for comprehensive radio radiance field reconstruction. RF-3DGS achieves highly accurate geometric information representation and exceptional rendering speed. This approach integrates both visual and radio radiance fields with encoded channel state information (CSI), providing a robust solution for advanced wireless communication and related applications.

## Requirements

### Hardware Requirements
- **GPU**: A CUDA-compatible GPU with at least 24 GB VRAM is recommended for radio spatial spectrum simulation and 3DGS training. If not available, adjust the simulation settings for lower memory consumption or use the pre-generated spectrum dataset provided in our repository. For lower VRAM setup, refer to the 3DGS documentation.

### Software Requirements
- **Operating System**: Ubuntu 22.04 LTS (recommended). Other versions may work but require compatibility checks for GPU drivers, CUDA, and PyTorch versions, as well as for building the 3DGS submodules.

- **Python 3.8** 

- **CUDA Toolkit**: Install CUDA Toolkit 11.x (11.7/11.8 tested) in the system environment (conda deactivate). Follow the [official installation guide](https://docs.nvidia.com/cuda/) for pre- and post-installation steps. While 12.x might work, it has not been tested.

- **Conda**: Highly recommended for environment setup.
```
conda create -n rf-3dgs python=3.8
conda activate rf-3dgs
```

- **Avoid 3DGS Repository Version Conflicts**: RF-3DGS is based on an earlier version of 3DGS. Use the following branch:  
  [3DGS commit b17ded92](https://github.com/graphdeco-inria/gaussian-splatting/tree/b17ded92b56ba02b6b7eaba2e66a2b0510f27764). Therefore, it is recommended to use an isolated environment specifically for RF-3DGS to avoid conflicts with other 3DGS projects.


## Simulation Tutorial 

RF-3DGS uses the [Sionna](https://github.com/NVlabs/sionna) library for radio spectrum simulation. Follow these steps to set up the simulation pipeline. (You can skip this part if use our RF-3DGS dataset.)

### Sionna Installation 

1. Clone the Sionna repository:
   ```bash
   git clone https://github.com/NVlabs/sionna.git
   ```
2. Install Docker and NVIDIA Container Toolkit:
   - Follow the [official Sionna installation guide](https://nvlabs.github.io/sionna/installation.html) to install docker and the NVIDIA Container Toolkit.
   - Installation guide of NVIDIA Container Toolkit: [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
3. Build the Sionna Docker image:
   ```bash
   cd sionna
   sudo make docker
   ```
4. Edit the `Makefile` to configure your Sionna docker project directory:
   Replace `{your/path}` with your desired directory path:
   ```
   run-docker:
       docker run -itd -v /your/path/:/tf/your_projects -u $(id -u):$(id -g) -p 8887:8888 --privileged=true $(GPU) --env NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility sionna
   ```
5. Run the Docker container:
   ```bash
   sudo make run-docker gpus=all
   ```
   Access the JupyterLab server at:  
   [`http://127.0.0.1:8887/lab/`](http://127.0.0.1:8887/lab/).

6. Copy the provided Jupyter notebook tutorials to your project directory `{your/path}` and start using them.
Download Jupyter Notebook at: 
[RF-3DGS Jupyter Notebook](https://drive.google.com/file/d/1OYgfRwIGhT05J8GlAr9U1zsX1yELzZ36/view?usp=sharing). 


## Dataset preparation

We provide our simulated radio spatial spectra for direct use with RF-3DGS: [RF-3DGS-Dataset](https://drive.google.com/file/d/1L6bbOWTlJyTPIz1oCuEJZuMr3T776hXm/view?usp=sharing). This dataset includes the following components:
- blender_visual_dataset: Contains the visual dataset for training the geometric representation.
- blender_visual_trained: Provides a pre-trained geometric model, allowing you to start training the Radiance Reconstruction Field (RRF) directly in the second stage.
- training_rf_spectrum: Contains the radio spatial spectra used for training the RRF.

### Visual Dataset
The visual dataset is generated in Blender using the Blender-NeRF add-on. Within our Blender scene file, a simple script is provided to generate the camera trajectory. You can then use the Blender-NeRF add-on to choose this camera trajectory and render images with precise camera poses.
For real-world images without known poses, you should follow the COLMAP pipeline to estimate the camera pose for each image before using them.


### RF Dataset Structure

1. **Data Format**:
RF-3DGS uses the COLMAP dataset format. If you want to use your custom RF data, you should organize them to the same structure. This includes:
- Spectra data `images/00xxx.png`
- Array poses `images.txt` and projection models `cameras.txt`
- `train_index.txt` and `test_index.txt` files for indexing, or set your own split method in dataset_readers.

2. **Folder Structure**:
   ```plaintext
   project_root/
   ├── images/
   │   ├── 00001.png
   │   ├── 00002.png
   │   └── ...
   ├── sparse/
   │   ├── 0/
   │       ├── cameras.txt
   │       ├── images.txt
   │       └── points3D.txt
   ├── train_index.txt
   ├── test_index.txt
---



## RF-3DGS Installation and Training

- Clone the RF-3DGS repository. 
   ```
   git clone <RF-3DGS-Repo-Link>
   cd RF-3DGS
   ```
- Install a compatible version of PyTorch from the [previous-versions](https://pytorch.org/get-started/previous-versions/) in the conda environment. For example, for CUDA 11.7 and conda, you can find this command:
   ```
   conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```

- Then install the following dependencies (later troubleshooting might be helpful for building the submodules):
   ```bash
   pip install plyfile
   conda install tqdm
   pip install submodules/diff-gaussian-rasterization
   pip install submodules/simple-knn
   ```
2. Train RF-3DGS (Start from visual trained checkpoint):
   ```bash
   python train.py -s RF-3DGS_dataset/training-rf-spectrum/3dgs_MPC_100 \
                   -m output/rf-3dgs_MPC_test \
                   --iterations 40000 \
                   --start_checkpoint "RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth" \
                   --eval \
                   --test_iterations 35000 40000 \
                   --save_iterations 40000
   ```
   For more training parameters, refer to the [3DGS documentation](https://github.com/graphdeco-inria/gaussian-splatting/tree/b17ded92b56ba02b6b7eaba2e66a2b0510f27764).

3. Optional: Training the Visual Geometry: To train the visual geometry, ensure you have an appropriate visual dataset or use our blender_visual_dataset. You can then follow the original 3DGS documentation to configure your training arguments. Additionally, set the --checkpoint_iterations parameter to specify the iteration from which the second RF training stage should commence. Here is a example:
  ```
  python train.py -s RF-3DGS_dataset/blender_visual_dataset/ \
                  -m output/rf-3dgs-visual \
                  --iterations 30000 \
                  --checkpoint_iterations 30000\
  ``` 

4. Optional: Install and run the SIBR Viewer to visualize trained models:
   ```bash
   sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
   cd SIBR_viewers
   cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j24 --target install
   ```
   View the trained model:
   ```bash
   SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output/3dgs_MPC_test
   ```

---

## Troubleshooting Submodule Builds

In most cases, following our guide should suffice for installing the 3DGS submodules and SIBR, as we have tested it on a workstation with various environment configurations without requiring additional steps. However, on another workstation, the only workable combination we tested is Ubuntu 22.04, NVIDIA 550 driver, CUDA 11.7 first installed in system environment, and then conda install cuda 11.7+ torch 1.13 in conda environment. 

There may be various issues when building the submodules. Therefore, it is essential to first examine the error messages carefully, such as unsupported compiler versions or other compatibility issues.

For CUDA-related errors, consider the following two solutions:

### Install CUDA Toolkit 11.7/11.8 in system environment

Ensure that CUDA Toolkit 11.7 or 11.8 is installed in the system environment (with Conda deactivated). 

Follow the official CUDA Toolkit installation guide or other tutorials for pre-installation steps. Below is an example of installing CUDA 11.7 on Ubuntu 22.04 using the NVIDIA repository and `apt-get`:

```
# Add the CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

# Update the system and install CUDA 11.7
sudo apt-get update
sudo apt install cuda-11-7

# Update PATH and LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda-11.7/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Update dynamic linker run-time bindings
sudo ldconfig
```
### Checking CUDA Availability in PyTorch

Verify the installation in the Conda environment by running the following commands:

Open Python in the Conda environment and run:
```
import torch
print(torch.cuda.is_available())
```

If torch.cuda.is_available() returns False, CUDA and PyTorch need to be reinstalled in the Conda environment.
Visit PyTorch's previous versions guide https://pytorch.org/get-started/previous-versions/ to find a compatible version of PyTorch and CUDA. Below is an example of installing PyTorch 1.13.1 with CUDA 11.7 in Conda:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```




