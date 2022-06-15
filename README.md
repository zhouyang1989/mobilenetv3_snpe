Step 1: Get a onnx file from the model mobilenetv3_small
"
  python train.py
"

Step 2: Convert the onnx to float dlc
"
  snpe-onnx-to-dlc -i ./mobilenetv3_small.onnx -o ./mobilenetv3_float.dlc --input_type "input_data" image --input_layout "input_data" NCHW
"

Step 3: Quantize the float dlc to quan dlc with hta
"
  snpe-dlc-quantize --input_dlc ./mobilenetv3_float.dlc --input_list ./raw_list.txt --output_dlc ./mobilenetv3_small_quan_hta.dlc --enable_hta
"
