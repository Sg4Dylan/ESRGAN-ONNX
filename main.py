import os
import numpy as np
from PIL import Image
import onnxruntime


class ESRGAN:

    def __init__(self, model_path):
        self.model_path = model_path
        self._init_model()


    def _img_preprocess(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            self.width, self.height = img.size
        except OSError:
            print(f'\nFile broken: {image_path}')
            return None
        input_data = np.array(img).transpose(2, 0, 1)
        img_data = input_data.astype('float32') / 255.0
        norm_img_data = img_data.reshape(1, 3, self.height, self.width).astype('float32')
        return norm_img_data


    def _init_model(self):
        self.session = onnxruntime.InferenceSession(self.model_path, None)
        self.model_input = self.session.get_inputs()[0].name
        return self.session, self.model_input


    def get_result(self, image_path):
        norm_img_data = self._img_preprocess(image_path)
        result = self.session.run([], {self.model_input: norm_img_data})[0][0]
        result = np.clip(result.transpose(1, 2, 0), 0, 1) * 255.0
        return result.round()

using_model_path = 'models/JPEG_Denoise/1x_JPEG_60-80-opti.onnx'
input_filename = 'input.jpg'
output_filename = 'output.png'
model = ESRGAN(using_model_path)
result = model.get_result(input_filename).astype(np.uint8)
new_im = Image.fromarray(result)
new_im.save(output_filename)
