# ESRGAN-ONNX

Implement of ESRGAN with ONNX, just only for inference.  
You can easily try your images without installing complex machine learning enviroment.

## Installation

1. Install dependence
 > pip install numpy pillow onnxruntime  

for NVIDIA GPU
 > pip install onnxruntime-gpu  

for AMD/Intel GPU, you could download and install `onnxruntime-dml` on [release page](https://github.com/Sg4Dylan/ESRGAN-ONNX/releases) or build it follow [this](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md)

2. Download `models.7z` on [release page](https://github.com/Sg4Dylan/ESRGAN-ONNX/releases)  
3. Unzip `models.7z` in code directory


## Testing

1. Modify source code  
```
# change model
using_model_path = 'models/JPEG_Denoise/1x_JPEG_60-80-opti.onnx'  
```

```
# change execution provider
self.exec_provider = 'CUDAExecutionProvider' # GPU via CUDA
self.exec_provider = 'DmlExecutionProvider'  # GPU via DirectML
self.exec_provider = 'CPUExecutionProvider'  # CPU Only
```

```
# set tile size
model = ESRGAN(using_model_path, tile_size=1024, scale=1)
```

2. Run to go
 > python main.py input.jpg

## Export others pretrain model
See this [gist](https://gist.github.com/Sg4Dylan/49d67f9b255e417d69dc19d97097982a)

## Reference
1. [ESRGAN](https://github.com/xinntao/ESRGAN)
2. [Model Database](https://upscale.wiki/wiki/Model_Database)