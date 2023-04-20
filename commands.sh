## For Debian

# Modify the /etc/apt/sources.list file and add at the end if not present, below is a sample, it might be different for you
deb http://deb.debian.org/debian/ bullseye main contrib non-free
deb-src http://deb.debian.org/debian/ bullseye main contrib non-free

deb http://security.debian.org/debian-security bullseye-security main contrib non-free
deb-src http://security.debian.org/debian-security bullseye-security main contrib non-free

deb http://deb.debian.org/debian/ bullseye-updates main contrib non-free
deb-src http://deb.debian.org/debian/ bullseye-updates main contrib non-free

## Method 1
# Search for the driver
apt-cache search nvidia-driver

# install Nvidia driver from official repo
sudo apt-get install nvidia-driver

## Method 2
# Alternatively you can install the nvidia-detect utility and get the best suitable driver for your system and card
sudo apt-get install nvidia-detect

# Now run this command to search for the best driver for your system and then install that
sudo nvidia-detect

# For latest generation cards, you might find that it is not able to find the driver for your card
# In this case you can install the driver using 1st method
# Once you are done till this step, it is better to reboot the system once

# Now run the command to see if your card is properly installed
nvidia-smi


# Now download Anaconda distribution
https://www.anaconda.com/products/distribution

# Download command (Depending on when you are watching the video, there might be a newer version)
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh

# Once downloaded, run the command
bash Anaconda3-2023.03-Linux-x86_64.sh

# Create env for TensorFlow and install
conda create -n tensorflow python=3.10 anaconda
conda activate tensorflow

# This might take a while
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# jupyter linking
conda install ipykernel
ipython kernel install --user --name=tensorflow-gpu

# # jupyter unlinking, only if you need to clean it later
# jupyter kernelspec list
# jupyter kernelspec uninstall tensorflow-gpu

## Env for PyTorch
conda create -n pytorch python=3.10 anaconda
conda activate pytorch

# Go to https://pytorch.org/get-started/locally/ and select the option best suited for your env
# Below is a sample command to install PyTorch with Cuda 11.8 using conda

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# jupyter linking
conda install ipykernel
ipython kernel install --user --name=pytorch-gpu

# # jupyter unlinking, only if you need to clean it later
# jupyter kernelspec list
# jupyter kernelspec uninstall pytorch-gpu

## Install CUDA on OS level

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-debian

# I prefer to install the repo manually
# Below commands are modified to use debian 11 as the distro, replace debian11 with your version accordingly

wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-debian11-keyring.gpg
sudo mv cuda-debian11-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-debian11-x86_64.list

wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-debian11.pin
sudo mv cuda-debian11.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-get update

# As of now the latest version of cuda supported by PyTorch and TensorFlow is 11.8
sudo apt-get install cuda-toolkit-11-8

# Check if the following directory is present

ls -l /usr/local/cuda

# If not present, then create a soft link from your cuda-11.8 directory to cuda
# This is will enable us to easy maintain cuda versions
ln -s /usr/local/cuda-11.8 /usr/local/cuda

# Download cudnn
# Go to https://developer.nvidia.com/cudnn and login to download the package, even though .deb package is available,
# I prefer to use the tar package
tar -xvf cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz

sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Now we need to add these to our ld-config path so that the libraries are available at login
cd /etc/ld.so.conf.d
sudo vi cuda.conf
# Add the below line
/usr/local/cuda/lib64

# Add the below to your path by adding the below lines to your .bashrc, .zshrc in /home/user directory
# This depends on the shell you use
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin


# Once done, run
sudo ldconfig

# If you find any issue, then reboot the system

nvcc test.cu -o test

# To further test GPU performance
g++ cpu.cpp -o cpu
nvcc gpu.cu -o gpu

time ./cpu
time ./gpu