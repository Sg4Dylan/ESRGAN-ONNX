import os
import sys
import numpy as np
from PIL import Image
import onnxruntime
import time


class ESRGAN:

    def __init__(self, model_path, tile_size=256, prepad=10, scale=4):
        self.model_path = model_path
        self.tile_size = tile_size
        self.prepad = prepad
        self.scale = scale
        self._init_model()


    def _tile_preprocess(self, img):
        input_data = np.array(img).transpose(2, 0, 1)
        img_data = input_data.astype('float32') / 255.0
        padded_size = self.tile_size + self.prepad*2
        norm_img_data = img_data.reshape(1, 3, padded_size, padded_size).astype('float32')
        return norm_img_data


    def _into_tiles(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            self.width, self.height = img.size
        except OSError:
            print(f'\nFile broken: {image_path}')
            return None
        tile_size = self.tile_size
        prepad = self.prepad
        self.num_width = int(np.ceil(self.width/tile_size))
        self.num_height = int(np.ceil(self.height/tile_size))
        self.pad_width = self.num_width*tile_size
        self.pad_height = self.num_height*tile_size
        pad_img = Image.new("RGB", (self.pad_width, self.pad_height))
        pad_img.paste(img)
        tiles = []
        for i in range(self.num_height):
            for j in range(self.num_width):
                box = [j*tile_size, i*tile_size, (j+1)*tile_size, (i+1)*tile_size]
                box = [box[0]-prepad, box[1]-prepad, box[2]+prepad, box[3]+prepad]
                tiles.append(self._tile_preprocess(pad_img.crop(tuple(box))))
        return tiles


    def _into_whole(self, tiles):
        scaled_tile = self.scale * self.tile_size
        scaled_pad = self.scale * self.prepad
        out_img = Image.new("RGB", (self.pad_width*self.scale, self.pad_height*self.scale))
        paste_cnt = 0
        for i in range(self.num_height):
            for j in range(self.num_width):
                box = (scaled_pad,scaled_pad,scaled_pad+scaled_tile,scaled_pad+scaled_tile)
                tile = tiles[paste_cnt].resize((scaled_pad*2+scaled_tile,scaled_pad*2+scaled_tile))
                paste_pos = (j*scaled_tile, i*scaled_tile)
                out_img.paste(tile.crop(box), paste_pos)
                paste_cnt += 1
        return out_img.crop((0, 0, self.width*self.scale, self.height*self.scale))


    def _init_model(self):
        self.exec_provider = 'DmlExecutionProvider' # 'CPUExecutionProvider' 'DmlExecutionProvider'
        self.session_opti = onnxruntime.SessionOptions()
        self.session_opti.enable_mem_pattern = False
        self.session = onnxruntime.InferenceSession(self.model_path, self.session_opti)
        self.session.set_providers([self.exec_provider])
        self.model_input = self.session.get_inputs()[0].name
        return self.session, self.model_input


    def get_result(self, image_path):
        input_tiles = self._into_tiles(image_path)
        output_tiles = []
        for tile in input_tiles:
            result = self.session.run([], {self.model_input: tile})[0][0]
            result = np.clip(result.transpose(1, 2, 0), 0, 1) * 255.0
            output_tiles.append(Image.fromarray(result.round().astype(np.uint8)))
        return self._into_whole(output_tiles)


using_model_path = 'models/JPEG_Denoise/1x_JPEG_60-80-opti.onnx'
input_path = sys.argv[1]
print('Init....')
model = ESRGAN(using_model_path, tile_size=1024, scale=1)
start_time = time.time()
result = model.get_result(input_path)
result.save(f'{input_path}.png')
print(f'Result saved. Time cost: {time.time()-start_time:.4f}s')
