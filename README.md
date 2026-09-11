# OCR demo:

- `ch_PP-OCRv4_det_infer.onnx`   – text detection
- `ch_ppocr_mobile_v2.0_cls_infer.onnx` – angle classification
- `en_PP-OCRv4_rec_infer.onnx`   – English recognition (ideal for ID cards)
- `en_dict.txt`

## Prerequisites (Mac M4)

```bash
brew install opencv cmake
# ONNX Runtime (arm64)
# https://github.com/microsoft/onnxruntime/releases
```

## Quick start

```bash
mkdir build && cd build
cmake .. \                 
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=~/work/libs/onnxruntime-osx-arm64-1.22.0
make -j$(sysctl -n hw.ncpu)

build/ocr_demo images/german-drivers-license.jpg
```

## Important note about the skeleton

The current source files contain **working model loading** and a compilable skeleton.
The heavy post-processing (DB unclip, CTC decoding, perspective crop) is intentionally left as placeholders.

For a **production-ready** version simply replace the four `src/*.cpp` + headers with the corresponding files from:

https://github.com/RapidAI/RapidOcrOnnx

Then re-run `cmake && make`. The CMakeLists and model layout already match the expected interface.

```bash
git clone https://github.com/RapidAI/RapidOcrOnnx.git
# Copy the full src/ and include/ from that repo into this project
# Keep the CMakeLists.txt and models/ we prepared
```