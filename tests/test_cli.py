"""Test cases for llm_chat.cli module."""

import argparse
import io
import sys
import re
import pytest
from contextlib import redirect_stderr

# Try to import with error handling - if import fails, skip all tests
try:
    from llm_chat.cli import (
        estimate_tokens,
        manage_context_efficiently,
        calculate_optimal_batch_size,
        suppress_specific_warning,
        parse_args,
    )
    IMPORT_SUCCESS = True
except ImportError:
    # Try alternative import paths
    try:
        import sys
        import os
        # Adjust the path to go up one directory from 'tests' to the project root, then into 'src'
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from llm_chat.cli import (
            estimate_tokens,
            manage_context_efficiently,
            calculate_optimal_batch_size,
            suppress_specific_warning,
            parse_args,
        )
        IMPORT_SUCCESS = True
    except ImportError:
        IMPORT_SUCCESS = False


# Skip all tests if imports fail
pytestmark = pytest.mark.skipif(
    not IMPORT_SUCCESS,
    reason="Cannot import required functions from llm_chat.cli. Ensure the 'src' directory is in PYTHONPATH or the package is installed."
)


def test_estimate_tokens_empty():
    """Test token estimation with empty messages."""
    messages = []
    assert estimate_tokens(messages) == 0


def test_estimate_tokens_nonempty():
    """Test token estimation with non-empty messages."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    total_chars = len("Hello") + len("Hi there!")
    expected = int(total_chars * 0.25)
    assert estimate_tokens(messages) == expected


def test_manage_context_efficiently_no_overflow():
    """Test context management when no overflow occurs."""
    # Create short messages so tokens < max_tokens * 0.75
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(3):
        messages.append({"role": "user", "content": "msg" + str(i)})
    result = manage_context_efficiently(messages, max_tokens=1000) # Ensure max_tokens is high enough
    assert result == messages  # No change since no overflow


def test_manage_context_efficiently_with_overflow():
    """Test context management when overflow occurs."""
    # Create messages with long content to force overflow
    system_msg = {"role": "system", "content": "System"}
    messages = [system_msg]
    # 10 user messages with long content
    for i in range(10):
        messages.append({"role": "user", "content": "x" * 200}) # Each message ~50 tokens
    # max_tokens small so overflow triggered (e.g., 10 * 50 = 500 tokens, if max_tokens is 100, 0.75*100=75, so 500 > 75)
    result = manage_context_efficiently(messages, max_tokens=100)
    # Should keep system_msg + last 6 user messages
    assert result[0] == system_msg
    assert len(result) == 1 + 6  # system + 6 most recent messages
    # Check that the content of the recent messages is correct
    for i in range(6):
        assert result[i+1]["content"] == messages[-6+i]["content"]


def test_calculate_optimal_batch_size_gpu():
    """Test batch size calculation for GPU usage."""
    # Based on pytest output, the cli.py being tested uses: min(2048, max(512, n_ctx // 3)) for GPU
    assert calculate_optimal_batch_size(n_ctx=3000, n_gpu_layers=1) == max(512, 3000 // 3) # 1000
    # Test the min cap
    assert calculate_optimal_batch_size(n_ctx=10000, n_gpu_layers=1) == min(2048, max(512, 10000 // 3)) # min(2048, 3333) = 2048
    # Test the max cap
    assert calculate_optimal_batch_size(n_ctx=1000, n_gpu_layers=1) == min(2048, max(512, 1000 // 3)) # min(2048, 512) = 512


def test_calculate_optimal_batch_size_cpu():
    """Test batch size calculation for CPU usage."""
    # When n_gpu_layers == 0, batch size is min(1024, max(256, n_ctx // 4))
    assert calculate_optimal_batch_size(n_ctx=1000, n_gpu_layers=0) == max(256, 1000 // 4) # 250 -> max(256,250) = 256
    assert calculate_optimal_batch_size(n_ctx=500, n_gpu_layers=0) == max(256, 500 // 4) # 125 -> max(256,125) = 256
    assert calculate_optimal_batch_size(n_ctx=10000, n_gpu_layers=0) == 1024  # capped at 1024 (10000//4 = 2500)


def test_suppress_specific_warning_filters_matching_lines():
    """Test warning suppression functionality."""
    warning_line = "llama_context: n_ctx_per_seq (512) < n_ctx_train (1024)\n"
    other_line = "Some other warning\n"
    pattern = r"llama_context: n_ctx_per_seq \(\d+\) < n_ctx_train \(\d+\)"

    buf = io.StringIO()
    # Replace sys.stderr with our buffer
    with redirect_stderr(buf):
        # Write both lines to stderr inside the context
        with suppress_specific_warning(pattern):
            sys.stderr.write(warning_line)
            sys.stderr.write(other_line)

    buf.seek(0)
    # Read all output at once to avoid issues with partial reads
    output_content = buf.getvalue()

    # The matching warning_line should be suppressed, only other_line appears
    assert other_line.strip() in output_content.strip()
    assert warning_line.strip() not in output_content.strip()


def test_parse_args_defaults(monkeypatch):
    """Test argument parsing with default values."""
    test_argv = [
        "prog_name_placeholder", # First element is program name
        "--model", "some/default-model-GGUF" # Required argument
    ]
    monkeypatch.setattr(sys, "argv", test_argv)
    args = parse_args()
    assert args.model == "some/default-model-GGUF"
    assert args.filename is None
    assert args.n_ctx == 4096
    assert args.n_gpu_layers == 0
    assert args.n_batch is None # Auto-calculated later
    assert isinstance(args.n_threads, int) # Default is multiprocessing.cpu_count() or 4
    assert args.flash_attn is False
    assert args.mul_mat_q is False
    assert args.use_mlock is False
    assert args.use_mmap is True
    assert args.n_keep == 64 # Default from cli.py being tested is 64
    assert abs(args.temperature - 0.7) < 1e-6
    assert abs(args.top_p - 0.9) < 1e-6
    assert args.profile is False


def test_parse_args_all_options(monkeypatch):
    """Test argument parsing with all options specified."""
    test_argv = [
        "prog_name_placeholder",
        "--model", "unsloth/Qwen2.5-Omni-3B-GGUF",
        "--filename", "*.gguf",
        "--n_ctx", "2048",
        "--n_gpu_layers", "2",
        "--n_batch", "128",
        "--n_threads", "8",
        "--flash_attn",
        "--mul_mat_q",
        "--use_mlock",
        # --use_mmap is True by default
        "--n_keep", "32",
        "--temperature", "0.5",
        "--top_p", "0.85",
        "--profile",
    ]
    monkeypatch.setattr(sys, "argv", test_argv)
    args = parse_args()
    assert args.model == "unsloth/Qwen2.5-Omni-3B-GGUF"
    assert args.filename == "*.gguf"
    assert args.n_ctx == 2048
    assert args.n_gpu_layers == 2
    assert args.n_batch == 128
    assert args.n_threads == 8
    assert args.flash_attn is True
    assert args.mul_mat_q is True
    assert args.use_mlock is True
    assert args.use_mmap is True
    assert args.n_keep == 32
    assert abs(args.temperature - 0.5) < 1e-6
    assert abs(args.top_p - 0.85) < 1e-6
    assert args.profile is True


def test_parse_args_with_hf_gguf_models(monkeypatch):
    """Test parsing with various Hugging Face GGUF model names."""
    test_cases = [
        "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "unsloth/Llama-3.1-8B-Instruct-GGUF",
        "TheBloke/phi-3-mini-4k-instruct-GGUF",
        "bartowski/Phi-3.1-mini-4k-instruct-GGUF",
        "microsoft/Phi-3-mini-4k-instruct-gguf",
    ]

    for model_name in test_cases:
        test_argv = ["prog_name_placeholder", "--model", model_name]
        monkeypatch.setattr(sys, "argv", test_argv)
        args = parse_args()
        assert args.model == model_name


def test_parse_args_filename_patterns_gguf(monkeypatch):
    """Test parsing with GGUF-specific filename patterns."""
    filename_patterns = [
        "*.gguf",
        "qwen2*.gguf",
        "model-q4_K_M.gguf",
        "*7b*.gguf",
    ]

    for pattern in filename_patterns:
        test_argv = [
            "prog_name_placeholder",
            "--model", "some-repo/some-model-GGUF", 
            "--filename", pattern
        ]
        monkeypatch.setattr(sys, "argv", test_argv)
        args = parse_args()
        assert args.filename == pattern

