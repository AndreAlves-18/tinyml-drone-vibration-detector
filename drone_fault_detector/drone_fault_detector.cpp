#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/i2c.h"

// TFLite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// modelo
#include "drone_fault_model.h"

// configurações do MPU6050
#define I2C_PORT       i2c0
#define I2C_SDA_PIN    4
#define I2C_SCL_PIN    5
#define MPU6050_ADDR   0x68
#define WINDOW_SIZE    50
#define N_AXES         3

// nomes das classes
const char* CLASS_NAMES[] = {"Normal", "Anomalo_N1", "Anomalo_N2"};

// memória para o TFLite Micro
constexpr int TENSOR_ARENA_SIZE = 60 * 1024;   // 60 KB
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// buffer da janela
float window_buffer[WINDOW_SIZE][N_AXES];

// parâmetros do Z-score copiar do Colab após treino
const float SCALER_MEAN[N_AXES]  = {-0.34582029f, -1.81494667f, 10.17652452f};
const float SCALER_SCALE[N_AXES] = { 7.15004794f,  6.99656644f,  4.80765432f};


// inicializa o MPU6050 
void mpu6050_init() {
    uint8_t buf[2];
    buf[0] = 0x6B; buf[1] = 0x00;
    i2c_write_blocking(I2C_PORT, MPU6050_ADDR, buf, 2, false);
    buf[0] = 0x1C; buf[1] = 0x00;
    i2c_write_blocking(I2C_PORT, MPU6050_ADDR, buf, 2, false);
}

// lê os 3 eixos de aceleração em m/s² 
void mpu6050_read_accel(float *ax, float *ay, float *az) {
    uint8_t reg = 0x3B;
    uint8_t raw[6];
    i2c_write_blocking(I2C_PORT, MPU6050_ADDR, &reg, 1, true);
    i2c_read_blocking(I2C_PORT, MPU6050_ADDR, raw, 6, false);

    int16_t ax_raw = (int16_t)((raw[0] << 8) | raw[1]);
    int16_t ay_raw = (int16_t)((raw[2] << 8) | raw[3]);
    int16_t az_raw = (int16_t)((raw[4] << 8) | raw[5]);

    // converte para m/s² (±2g → divisor 16384, multiplicado por 9.81)
    *ax = (ax_raw / 16384.0f) * 9.81f;
    *ay = (ay_raw / 16384.0f) * 9.81f;
    *az = (az_raw / 16384.0f) * 9.81f;
}

// normaliza uma janela com Z-score 
void normalize_window(float input[WINDOW_SIZE][N_AXES],
                      float output[WINDOW_SIZE][N_AXES]) {
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int c = 0; c < N_AXES; c++) {
            output[t][c] = (input[t][c] - SCALER_MEAN[c]) / SCALER_SCALE[c];
        }
    }
}



int main() {
    stdio_init_all();
    sleep_ms(2000);   // aguarda USB estabilizar
    printf("=== Drone Fault Detector ===\n");

    // inicializa I2C
    i2c_init(I2C_PORT, 400 * 1000);   // 400 kHz
    gpio_set_function(I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_PIN);
    gpio_pull_up(I2C_SCL_PIN);
    mpu6050_init();
    printf("MPU6050 inicializado\n");

    // carrega o modelo TFLite
    const tflite::Model* tfl_model = tflite::GetModel(g_model);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERRO: versão do modelo incompatível!\n");
        while (1);
    }

    // registra apenas as operações que o modelo usa
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddMean();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddQuantize();
    resolver.AddExpandDims();
    resolver.AddDequantize();

    static tflite::MicroInterpreter interpreter(
        tfl_model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERRO: AllocateTensors falhou!\n");
        while (1);
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("Modelo carregado. Iniciando coleta...\n\n");

    // antes do loop principal
    uint32_t t_start = time_us_32();

    for (int i = 0; i < WINDOW_SIZE; i++) {
        mpu6050_read_accel(
            &window_buffer[i][0],
            &window_buffer[i][1],
            &window_buffer[i][2]
        );
        sleep_ms(10);
    }

    uint32_t t_end = time_us_32();
    float fs_real = WINDOW_SIZE / ((t_end - t_start) / 1e6f);
    printf("fs real: %.1f Hz\n", fs_real);

    // loop principal
    while (true) {

        // coleta janela de tamanho  WINDOW_SIZE
        for (int i = 0; i < WINDOW_SIZE; i++) {
            mpu6050_read_accel(
                &window_buffer[i][0],
                &window_buffer[i][1],
                &window_buffer[i][2]
            );
            sleep_ms(10);   // ~100 Hz
        }

        // normaliza
        float norm_buffer[WINDOW_SIZE][N_AXES];
        normalize_window(window_buffer, norm_buffer);

        // copia para o modelo
        float* input_data = input->data.f;
        for (int t = 0; t < WINDOW_SIZE; t++) {
            for (int c = 0; c < N_AXES; c++) {
                input_data[t * N_AXES + c] = norm_buffer[t][c];
            }
        }

        // inferência
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("ERRO: Invoke falhou!\n");
            continue;
        }

        // lê probabilidades e acha a classe
        float* probs = output->data.f;
        int best = 0;
        for (int i = 1; i < 3; i++) {
            if (probs[i] > probs[best]) best = i;
        }

        printf("Resultado: %-12s  (N:%.2f  N1:%.2f  N2:%.2f)\n",
               CLASS_NAMES[best], probs[0], probs[1], probs[2]);
    }

    return 0;
}