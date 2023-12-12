# YOLOX Tensorrt Quantization

## Introduction
Converting YOLOX model(onnx format) to engine type based on tensorrt. 

Supporting fp16/int8 mode.

## Data Preparation
When quantize the onnx model to int8 engine model, firstly, we need to prepare the calibration dataset.
* In this repo, you can run the sample.py:
    ```shell
    python sample.py --training_data_path "your_training_data" --count "num_imgs_sample" --calibration_path ./calibration/
    ```

## Model Converting
* 1、Converting to fp16 engine model
    ```shell
    python quantization.py --onnx_file_path "onnx_model_path" --engine_file_path "converted_engine_path" --batch_size 1 --mode fp16 
    ```
* 2、Converting to int8 engine model
    ```shell
    python quantization.py --onnx_file_path "onnx_model_path" --engine_file_path "converted_engine_path" --batch_size 1 --mode int8 
    ```
*  Notes: if you want to convert multi-batch model, firstly, you need to embrace an onnx model with dynamic batch. Then, convert it to int8 engine model with specific batch_size.