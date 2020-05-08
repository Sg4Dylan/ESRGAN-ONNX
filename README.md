# ESRGAN-ONNX

Implement of ESRGAN with ONNX, just only for inference.  
You can easily try your images without installing complex machine learning enviroment.

## Installation

1. Install dependence
 > pip install numpy pillow onnxruntime

2. Download `models.7z` in [release](https://github.com/Sg4Dylan/ESRGAN-ONNX/releases) page
3. Unzip `models.7z` in code directory


## Testing

1. Modify source code  
```
# main.py #L39~#L41
using_model_path = 'models/JPEG_Denoise/1x_JPEG_60-80-opti.onnx'  
input_filename = 'YOUR_INPUT_IMAGE_FILE_PATH'  
output_filename = 'YOUR_OUTPUT_IMAGE_FILE_PATH'
```

2. Run to go
 > python main.py

## Export others pretrain model
See this [gist](https://gist.github.com/Sg4Dylan/49d67f9b255e417d69dc19d97097982a)

## Reference
1. [ESRGAN](https://github.com/xinntao/ESRGAN)
2. [Model Database](https://upscale.wiki/wiki/Model_Database)