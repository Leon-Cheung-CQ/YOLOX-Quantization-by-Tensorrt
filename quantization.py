import argparse

import cv2
import os
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit


class yoloxEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, imgpath, batch_size, channel, inputsize=[640, 640]):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = 'yolox.cache'
        self.batch_size = batch_size
        self.Channel = channel
        self.height = inputsize[0]
        self.width = inputsize[1]
        self.imgs = [os.path.join(imgpath, file) for file in os.listdir(imgpath) if file.endswith('jpg')]
        np.random.shuffle(self.imgs)
        self.imgs = self.imgs[:2000]
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs) // self.batch_size
        self.calibration_data = np.zeros((self.batch_size, 3, self.height, self.width), dtype=np.float32)
        self.data_size = self.calibration_data.nbytes
        self.device_input = cuda.mem_alloc(self.data_size)

    def free(self):
        self.device_input.free()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.Channel * self.height * self.width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs)
            return [self.device_input]
        except:
            print("wrong")
            return None

    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.height, self.width), dtype=np.float32)

            for i, f in enumerate(batch_files):
                img = cv2.imread(f)
                img = cv2.resize(img, (self.height, self.width))
                img = img.transpose((2, 0, 1))[::-1, :, :]
                img = np.ascontiguousarray(img)
                img = img.astype(np.float32) / 255.
                assert (img.nbytes == self.data_size / self. batch_size), "not valid img!" + f
                batch_imgs[i] = img
            self.batch_idx += 1
            print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()


def get_engine(onnx_file_path, engine_file_path, cali_img, batch_size, mode='FP32', workspace_size=4096*10):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def build_engine():
        assert mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                explicit_batch_flag
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser:
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            config.max_workspace_size = workspace_size * (1024 * 1024)  # workspace_sizeMiB
            # 构建精度
            if mode.lower() == 'fp16':
                config.flags |= 1 << int(trt.BuilderFlag.FP16)

            if mode.lower() == 'int8':
                print('trt.DataType.INT8')
                config.flags |= 1 << int(trt.BuilderFlag.INT8)
                config.flags |= 1 << int(trt.BuilderFlag.FP16)
                calibrator = yoloxEntropyCalibrator(cali_img, batch_size, 3, [640, 640])
                config.int8_calibrator = calibrator

            profile = builder.create_optimization_profile()
            profile.set_shape(
                network.get_input(0).name,
                min=(1, 3, 640, 640),
                opt=(max(1, batch_size//2), 3, 640, 640),
                max=(batch_size, 3, 640, 640)
            )
            config.add_optimization_profile(profile)
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main(onnx_file_path, engine_file_path, cali_img_path, batch_size, mode='FP32'):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    get_engine(onnx_file_path, engine_file_path, cali_img_path, batch_size, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file_path', type=str, required=True, help='onnx model path')
    parser.add_argument('--engine_file_path', type=str, required=True, help='converted engine path')
    parser.add_argument('--calibration_path', type=str, default="./calibration/", help='calibration image path')
    parser.add_argument('--batch_size', type=int, help='quantization dataloader batch size')
    parser.add_argument('--mode', type=str, default='int8', help='quantization type (fp16/int8)')
    args = parser.parse_args()
    main(args.onnx_file_path, args.engine_file_path, args.calibration_path, args.batch_size, args.mode)


