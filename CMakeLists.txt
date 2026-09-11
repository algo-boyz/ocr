cmake_minimum_required(VERSION 3.16)
project(PaddleOCR_ONNX_CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force Apple Silicon when on macOS
if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

# Dependencies
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "$ENV{HOME}/work/libs/onnxruntime-osx-arm64-1.22.0" CACHE PATH "")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(ocr_demo
  src/main.cpp
  src/ocr.cpp
  src/db.cpp
  src/angle.cpp
  src/crnn.cpp
  src/utils.cpp
)

target_link_libraries(ocr_demo
  ${OpenCV_LIBS}
  onnxruntime
)

# Copy models next to the binary for convenience
add_custom_command(TARGET ocr_demo POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:ocr_demo>/models
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models $<TARGET_FILE_DIR:ocr_demo>/models
  COMMENT "Copying models to build directory"
)

message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
