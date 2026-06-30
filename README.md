# Detecção de Falhas em Hélices de Drone com TinyML

Sistema de detecção de falhas em hélices de drone baseado em sinais de vibração, utilizando uma CNN 1D treinada em Python/Keras e implantada em um microcontrolador Raspberry Pi Pico W via TensorFlow Lite Micro.

## Sobre o projeto

O objetivo é identificar três condições de operação a partir de sinais de vibração capturados por um acelerômetro (MPU6050):

- **Normal** — hélice balanceada
- **Anômalo N1** — desbalanceamento leve (menos fita adesiva)
- **Anômalo N2** — desbalanceamento severo (mais fita adesiva)

O pipeline completo cobre desde a coleta e pré-processamento dos dados até o treinamento do modelo e a inferência embarcada em tempo real no microcontrolador, sem depender de processamento externo ou conexão com a nuvem.

Este projeto foi desenvolvido como parte do TCC e da disciplina de Aprendizado de Máquina Embarcado (AME) na Universidade Federal do Ceará (UFC), Campus Quixadá.

## Estrutura do repositório

```
.
├── colab/
│   └── treinamento_modelo.ipynb       # notebook com todo o pipeline de ML
├── firmware/
│   ├── drone_fault_detector.cpp       # código principal do firmware
│   ├── drone_fault_model.cc           # modelo convertido em array C
│   ├── drone_fault_model.h            # header do modelo
│   ├── CMakeLists.txt                 # configuração de build
│   └── pico_sdk_import.cmake
└── README.md
```

## Pipeline de Machine Learning (Colab)

O notebook cobre as seguintes etapas:

1. **Carregamento dos dados** — leitura dos arquivos `.xlsx` contendo os sinais de vibração brutos (eixos X, Y, Z)
2. **Exploração dos dados** — estatísticas descritivas, visualização temporal e verificação de balanceamento entre classes
3. **Pré-processamento** — segmentação do sinal contínuo em janelas de 26 amostras e normalização Z-score
4. **Divisão temporal** — separação em treino (70%), validação (15%) e teste (15%) respeitando a ordem cronológica das janelas, evitando vazamento de dados entre conjuntos
5. **Modelo** — CNN 1D compacta (TinyCNN1D), com blocos Conv1D → BatchNorm → ReLU → MaxPooling, finalizando em GlobalAveragePooling e uma camada densa com Softmax
6. **Treinamento** — otimizador Adam, *categorical crossentropy*, com EarlyStopping, ReduceLROnPlateau e ModelCheckpoint
7. **Avaliação** — acurácia, precisão, recall, F1-score e matriz de confusão no conjunto de teste
8. **Conversão para TinyML** — exportação do modelo para TensorFlow Lite com quantização int8, seguida da geração do array C (`.cc`) compatível com TensorFlow Lite Micro

### Tecnologias utilizadas

- Python 3.12
- TensorFlow / Keras
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Google Colab

## Firmware embarcado (Raspberry Pi Pico W)

O firmware foi desenvolvido em C/C++ utilizando o Raspberry Pi Pico SDK e a biblioteca [pico-tflmicro](https://github.com/raspberrypi/pico-tflmicro), responsável por executar a inferência do modelo TFLite diretamente no microcontrolador.

### Funcionamento

1. Inicializa a comunicação I2C com o acelerômetro MPU6050
2. Coleta uma janela de 26 amostras dos três eixos de aceleração
3. Normaliza os dados coletados
4. Executa a inferência com o TensorFlow Lite Micro
5. Identifica a classe com maior probabilidade e imprime o resultado via USB serial

### Hardware utilizado

- Raspberry Pi Pico W (RP2040)
- Acelerômetro MPU6050 (comunicação I2C)

### Build do projeto

O projeto utiliza CMake e Ninja para compilação:

```bash
mkdir build
cd build
cmake ..
ninja
```

Após a compilação, grave o arquivo `.uf2` gerado no Pico (modo BOOTSEL):

```bash
cp drone_fault_detector.uf2 /media/$USER/RPI-RP2/
```

Para visualizar os resultados da inferência em tempo real, conecte-se à porta serial USB do Pico (baud rate 115200), por exemplo via `minicom`:

```bash
minicom -b 115200 -o -D /dev/ttyACM0
```

### Dependências

- [Raspberry Pi Pico SDK](https://github.com/raspberrypi/pico-sdk)
- [pico-tflmicro](https://github.com/raspberrypi/pico-tflmicro)

## Dataset

O modelo foi treinado com um dataset próprio, coletado em bancada com o acelerômetro MPU6050 acoplado a um motor de drone, cobrindo as três condições de operação descritas acima. A metodologia de simulação de falhas (adição de fita adesiva na hélice) foi inspirada no protocolo de Al-Haddad et al. (2025).

## Autor

André Alves de Freitas — Engenharia de Computação, UFC Campus Quixadá
Orientador: Jeandro de Mesquita Bezerra
