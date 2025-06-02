# LLM-Chat

![CI](https://github.com/rice8y/llm-chat/actions/workflows/CI.yml/badge.svg)
![codecov](https://codecov.io/gh/rice8y/llm-chat/branch/main/graph/badge.svg)
![License](https://img.shields.io/github/license/rice8y/llm-chat.svg)

**LLM-Chat** is a high-performance command-line chat tool built on top of [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), a Python binding for llama.cpp. It allows you to run and interact with local LLMs (such as LLaMA, Gemma, Mistral, etc.) from your terminal with advanced optimization features and minimal setup.

## Features

- **GPU Acceleration**: Full support for CUDA and Metal GPU acceleration
- **Advanced Optimization**: Flash Attention, quantized matrix multiplication, and memory optimization
- **Intelligent Context Management**: Automatic context window management to prevent overflow
- **Performance Profiling**: Real-time token generation speed monitoring
- **Memory Optimization**: Memory locking and mapping for improved performance
- **Dynamic Configuration**: Auto-calculated optimal batch sizes based on hardware
- **Rich Console Interface**: Beautiful terminal UI with spinners and progress indicators

## Installation

You can install this CLI tool using `uv` in two different ways:

### A. Install directly from GitHub (recommended)

```bash
uv tool install git+https://github.com/rice8y/llm-chat.git
```

This will fetch and install the latest version directly from the repository.

### B. Install from a local clone

1. Clone the repository:

```bash
git clone https://github.com/rice8y/llm-chat.git
```

2. Move into the project directory:

```bash
cd llm-chat
```

3. Install the package in editable mode using `uv tool`:

```bash
uv tool install -e .
```

This is useful if you plan to modify the code locally.

## GPU Support

For optimal performance with GPU acceleration, ensure llama-cpp-python is compiled with appropriate GPU support:

### NVIDIA CUDA

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

### NVIDIA CUDA with Flash Attention

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DLLAMA_FLASH_ATTN=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

### Apple Metal (Apple Silicon)

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

## Usage

```bash
llm-chat [-h] --model MODEL [options]
```

### Core Options

- `--model MODEL`: Path to the Llama model file (gguf format) or Hugging Face repo ID
- `--filename FILENAME`: Filename pattern for model file when using Hugging Face Hub
- `--n_ctx N_CTX`: Context size for the model (default: 4096)

### Performance Options

- `--n_gpu_layers N_GPU_LAYERS`: Number of layers to offload to GPU (-1 for all layers, default: 0)
- `--n_batch N_BATCH`: Batch size for prompt processing (auto-calculated if not specified)
- `--n_threads N_THREADS`: Number of CPU threads to use (default: CPU count)
- `--flash_attn`: Enable Flash Attention if supported by build and hardware
- `--mul_mat_q`: Enable quantized matrix multiplication for potential speedup

### Memory Optimization Options

- `--use_mlock`: Lock model in memory to prevent swapping
- `--use_mmap`: Use memory mapping for faster file I/O (default: enabled)
- `--n_keep N_KEEP`: Number of tokens to keep in cache (default: 64)

### Generation Options

- `--temperature TEMPERATURE`: Sampling temperature (default: 0.7)
- `--top_p TOP_P`: Top-p sampling parameter (default: 0.9)

### Utility Options

- `--profile`: Enable performance profiling with token generation speed
- `-h`, `--help`: Show help message and exit

## Examples

### Basic Usage

```bash
# Local model with default settings
llm-chat --model /path/to/Llama-3.2-1B-Instruct-F16.gguf

# Hugging Face model
llm-chat --model unsloth/gemma-3-4b-it-GGUF --filename gemma-3-4b-it-UD-Q8_K_XL.gguf
```

### High-Performance GPU Setup

```bash
# Maximum GPU acceleration with all optimizations
llm-chat --model /path/to/model.gguf \
  --n_gpu_layers -1 \
  --flash_attn \
  --mul_mat_q \
  --use_mlock \
  --profile

# Large context with custom settings
llm-chat --model /path/to/model.gguf \
  --n_ctx 8192 \
  --n_gpu_layers 32 \
  --n_batch 1024 \
  --temperature 0.5
```

### CPU-Optimized Setup

```bash
# Optimized for CPU-only usage
llm-chat --model /path/to/model.gguf \
  --n_threads 16 \
  --n_batch 512 \
  --use_mlock \
  --profile
```

### Memory-Constrained Environment

```bash
# Reduced memory usage
llm-chat --model /path/to/model.gguf \
  --n_ctx 2048 \
  --n_batch 256 \
  --n_keep 32
```

## Chat Commands

During chat sessions, you can use the following commands:

- `exit` or `quit`: End the chat session
- `clear`: Clear chat history while keeping the session active

## Performance Tips

1. **Model Quantization**: Use more aggressive quantization (Q4_K_M, Q4_K_S) for better performance
2. **GPU Memory**: Increase `n_gpu_layers` gradually to find optimal GPU/CPU balance
3. **Batch Size**: Larger batch sizes generally improve throughput for longer prompts
4. **Context Management**: The tool automatically manages context to prevent overflow
5. **Memory Locking**: Use `--use_mlock` on systems with sufficient RAM to prevent swapping

## Troubleshooting

### GPU Issues

- Ensure CUDA toolkit is properly installed for NVIDIA GPUs
- Verify llama-cpp-python compilation with GPU support
- Check GPU memory usage and reduce `n_gpu_layers` if needed

### Memory Issues

- Reduce `n_ctx` or `n_batch` for memory-constrained systems
- Use more aggressive model quantization
- Enable automatic context management (enabled by default)

### Performance Issues

- Enable profiling with `--profile` to identify bottlenecks
- Experiment with different batch sizes and thread counts
- Consider model quantization vs. accuracy trade-offs

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).