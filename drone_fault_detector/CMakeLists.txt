cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(drone_fault_detector C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

# inclui o TFLite Micro
set(PICO_SDK_PATH "/home/andre/.pico-sdk/sdk/2.2.0")
add_subdirectory(pico-tflmicro)

add_executable(drone_fault_detector
    drone_fault_detector.cpp
    drone_fault_model.cc          # modelo convertido
)

target_link_libraries(drone_fault_detector
    pico_stdlib
    hardware_i2c
    pico-tflmicro                 # biblioteca TFLite Micro
)

# habilita saída USB para prints de debug
pico_enable_stdio_usb(drone_fault_detector 1)
pico_enable_stdio_uart(drone_fault_detector 0)

# gera o arquivo .uf2 para upload no Pico
pico_add_extra_outputs(drone_fault_detector)