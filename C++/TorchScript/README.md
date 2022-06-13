# Torch script C++

## Install Torch script  
### Download Torch script  
- LTS(1.8.2)
- Linux
- LibTorch
- C++
- CUDA 11.0

``` bash 
wget https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu111.zip
```

### Unpack Torch script .zip 
``` bash
unzip libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu111.zip
```

## Install OpenCV

### Download OpenCV
```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
```

### Unpack OpenCV
```bash
unzip opencv.zip
unzip opencv_contrib.zip
```

### Make directory for build
```bash
mkdir -p build && cd build
```

### Configure
```bash
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
```

### Build OpenCV
```bash
make -j 8
```