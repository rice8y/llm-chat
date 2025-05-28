# LLM-Chat

**LLM-Chat** is a simple command-line chat tool built on top of [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), a Python binding for llama.cpp.

It allows you to run and interact with local LLMs (such as LLaMA, Gemma, Mistral, etc.) from your terminal with minimal setup.

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

## Usage

```bash
llm-chat [-h] --model MODEL [--filename FILENAME] [--n_ctx N_CTX]
```

### Options

- `-h`, `--help`: show this help message and exit
- `--model MODEL`: Path to the Llama model file.
- `--filename FILENAME`: Filename pattern for the model file.
- `--n_ctx N_CTX`: Context size for the model.

## Examples

### Using local models

```bash
llm-chat --model /path/to/Llama-3.2-1B-Instruct-F16.gguf
```

### Pulling models from Hugging Face Hub

```bash
llm-chat --model unsloth/gemma-3-4b-it-GGUF --filename gemma-3-4b-it-UD-Q8_K_XL.gguf
```

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).