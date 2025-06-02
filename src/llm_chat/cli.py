import argparse
import asyncio
import io
import multiprocessing
import os
import re
import sys
import time
from contextlib import contextmanager

from llama_cpp import Llama
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner as RichSpinner
from rich.text import Text

console = Console()

@contextmanager
def suppress_specific_warning(pattern):
    old_stderr = sys.stderr
    stderr_buffer = io.StringIO()
    sys.stderr = stderr_buffer
    try:
        yield
    finally:
        sys.stderr = old_stderr
        stderr_buffer.seek(0)
        try:
            regex = re.compile(pattern)
        except re.error as e:
            console.print(f"[bold red]Regex error in suppress_specific_warning:[/bold red] {e}")
            for line in stderr_buffer:
                sys.stderr.write(line)
            return

        for line in stderr_buffer:
            if not regex.search(line):
                sys.stderr.write(line)

def estimate_tokens(messages: list[dict[str, str]]) -> int:
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    return int(total_chars * 0.25)

def manage_context_efficiently(messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    if estimate_tokens(messages) > max_tokens * 0.75:
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        recent_messages = messages[-6:] if len(messages) > 6 else messages[1:]

        if system_msg:
            return [system_msg] + recent_messages
        return recent_messages
    return messages

def calculate_optimal_batch_size(n_ctx: int, n_gpu_layers: int) -> int:
    if n_gpu_layers > 0:
        return min(2048, max(512, n_ctx // 3))
    else:
        return min(1024, max(256, n_ctx // 4))

def profile_operation(operation_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            console.print(f"[dim]{operation_name}: {end_time - start_time:.3f}s[/dim]")
            return result
        return wrapper
    return decorator

def parse_args():
    parser = argparse.ArgumentParser(description="High-performance interactive chat with LLM")
    parser.add_argument("--model", type=str, required=True, help="Path to the Llama model file (gguf format)")
    parser.add_argument("--filename", type=str, default=None,
                        help="Filename pattern for the model file if downloading from Hugging Face Hub")
    parser.add_argument("--n_ctx", type=int, default=4096, help="Context size for the model")
    parser.add_argument("--n_gpu_layers", type=int, default=0,
                        help="Number of layers to offload to GPU (-1 for all)")
    parser.add_argument("--n_batch", type=int, default=None,
                        help="Batch size for prompt processing (auto-calculated if not specified)")

    try:
        default_threads = multiprocessing.cpu_count()
    except NotImplementedError:
        default_threads = 4

    parser.add_argument("--n_threads", type=int, default=default_threads,
                        help="Number of CPU threads to use")
    parser.add_argument("--flash_attn", action="store_true",
                        help="Enable Flash Attention if supported")
    parser.add_argument("--mul_mat_q", action="store_true",
                        help="Enable quantized matrix multiplication")
    parser.add_argument("--use_mlock", action="store_true",
                        help="Lock model in memory to prevent swapping")
    parser.add_argument("--use_mmap", action="store_true", default=True,
                        help="Use memory mapping for faster file I/O")
    parser.add_argument("--n_keep", type=int, default=64,
                        help="Number of tokens to keep in cache")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--profile", action="store_true",
                        help="Enable performance profiling")

    return parser.parse_args()

def show_spinner(message, func, *args, **kwargs):
    spinner = RichSpinner("dots")
    message_text = Text(message)
    live_display = Columns([spinner, message_text], expand=False)
    try:
        with Live(live_display, refresh_per_second=10, transient=True, console=console):
            result = func(*args, **kwargs)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation interrupted by user.[/bold yellow]")
        sys.exit(0)
    return result

@profile_operation("Model loading")
def load_model(args):
    warning_pattern_n_ctx = r"llama_context: n_ctx_per_seq \(\d+\) < n_ctx_train \(\d+\)"

    if args.n_batch is None:
        args.n_batch = calculate_optimal_batch_size(args.n_ctx, args.n_gpu_layers)
        console.print(f"[dim]Auto-calculated batch size: {args.n_batch}[/dim]")

    llama_cpp_kwargs = {
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "n_batch": args.n_batch,
        "n_threads": args.n_threads if args.n_gpu_layers == 0 else None,
        "verbose": False,
        "use_mlock": args.use_mlock,
        "use_mmap": args.use_mmap,
        "n_keep": args.n_keep,
    }

    if args.flash_attn:
        llama_cpp_kwargs["flash_attn"] = True
        console.print("[dim]Flash Attention enabled[/dim]")

    if args.mul_mat_q:
        llama_cpp_kwargs["mul_mat_q"] = True
        console.print("[dim]Quantized matrix multiplication enabled[/dim]")

    with suppress_specific_warning(warning_pattern_n_ctx):
        if os.path.exists(args.model):
            return Llama(model_path=args.model, **llama_cpp_kwargs)
        elif args.filename:
            return Llama.from_pretrained(
                repo_id=args.model,
                filename=args.filename,
                **llama_cpp_kwargs
            )
        else:
            console.print(f"[bold yellow]Warning:[/bold yellow] Loading from repo '{args.model}' without filename pattern")
            return Llama.from_pretrained(repo_id=args.model, **llama_cpp_kwargs)

async def async_generate_response(model, messages, temperature, top_p):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: model.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=temperature,
            top_p=top_p
        )
    )

def get_user_input(prompt: str, use_gpu: bool) -> str:
    try:
        if use_gpu:
            return input(prompt)
        else:
            return console.input(prompt)
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt from None

def handle_context_overflow(messages: list[dict[str, str]], args) -> list[dict[str, str]]:
    console.print(f"\n[bold red]Context limit reached ({args.n_ctx} tokens)[/bold red]")
    console.print("[yellow]Options:[/yellow]")
    console.print("1. Clear history and continue (c)")
    console.print("2. Reduce context automatically (r)")
    console.print("3. Exit (e)")

    try:
        choice = get_user_input("Choose option [c/r/e]: ", args.n_gpu_layers != 0).strip().lower()
    except KeyboardInterrupt:
        return None

    if choice == "c":
        return [msg for msg in messages if msg.get("role") == "system"]
    elif choice == "r":
        return manage_context_efficiently(messages, args.n_ctx)
    else:
        return None

def main():
    args = parse_args()

    if args.n_gpu_layers != 0:
        console.print("[bold cyan]GPU acceleration enabled[/bold cyan]")
        console.print("Ensure llama-cpp-python is compiled with GPU support")
    else:
        console.print("[bold yellow]Running on CPU[/bold yellow]")
        console.print(f"Using {args.n_threads} CPU threads")

    try:
        model = show_spinner("Loading model...", load_model, args)
    except ValueError as e:
        console.print(f"\n[bold red]Error loading model:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error during model loading:[/bold red] {e}")
        if "DLL load failed" in str(e) or "module not found" in str(e).lower():
            console.print("[bold yellow]Hint:[/bold yellow] Missing CUDA toolkit or incorrect installation")
        sys.exit(1)

    console.clear()
    console.print("\n[bold green]Model loaded successfully![/bold green]")

    config_info = [
        f"Context: {args.n_ctx}",
        f"GPU Layers: {args.n_gpu_layers}",
        f"Batch Size: {args.n_batch}",
        f"CPU Threads: {args.n_threads or 'Auto'}",
        f"Flash Attn: {args.flash_attn}",
        f"MulMatQ: {args.mul_mat_q}",
        f"Memory Lock: {args.use_mlock}",
        f"Memory Map: {args.use_mmap}"
    ]

    rprint(f"[dim]{' | '.join(config_info)}[/dim]")
    console.print("[dim]Enter 'exit', 'quit', or 'clear' to manage chat[/dim]\n")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    try:
        while True:
            try:
                user_input = get_user_input(">>> ", args.n_gpu_layers != 0)
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Exiting chat[/bold yellow]")
                break

            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold yellow]Exiting chat[/bold yellow]")
                break

            if user_input.lower() == "clear":
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                console.print("[bold yellow]Chat history cleared[/bold yellow]\n")
                continue

            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})
            messages = manage_context_efficiently(messages, args.n_ctx)

            try:
                spinner = RichSpinner("dots")
                reply_content = ""
                live_display = Columns([spinner, Text("Thinking...")], expand=False)
                first_chunk_received = False

                def get_response(msgs):
                    return model.create_chat_completion(
                        messages=msgs,
                        stream=True,
                        temperature=args.temperature,
                        top_p=args.top_p
                    )

                generation_start = time.perf_counter() if args.profile else None

                with Live(live_display, refresh_per_second=10, transient=True, console=console) as live:
                    response_stream = get_response(messages)
                    for chunk in response_stream:
                        if not first_chunk_received:
                            live.stop()
                            first_chunk_received = True

                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        content_piece = delta.get("content")

                        if content_piece:
                            console.print(content_piece, end="", soft_wrap=True, highlight=False)
                            reply_content += content_piece

                        if choice.get("finish_reason") is not None:
                            break

                if args.profile and generation_start:
                    generation_time = time.perf_counter() - generation_start
                    token_count = len(reply_content.split())
                    tokens_per_second = token_count / generation_time if generation_time > 0 else 0
                    console.print(f"\n[dim]Generation: {generation_time:.3f}s | Tokens: {token_count} | Speed: {tokens_per_second:.1f} tok/s[/dim]")

                if reply_content:
                    console.print()
                    messages.append({"role": "assistant", "content": reply_content})
                elif not first_chunk_received:
                    console.print("[bold red]\nNo response generated[/bold red]")
                elif first_chunk_received and not reply_content:
                    console.print("[bold red]\nEmpty response received[/bold red]")

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Generation interrupted[/bold yellow]")
                if reply_content:
                    messages.append({"role": "assistant", "content": reply_content + " [Interrupted]"})
                continue
            except Exception as e:
                error_msg = str(e)
                if any(keyword in error_msg for keyword in ["exceed context", "failed to eval", "llama_decode"]):
                    new_messages = handle_context_overflow(messages, args)
                    if new_messages is None:
                        console.print("[bold yellow]Exiting chat[/bold yellow]")
                        break
                    messages = new_messages
                    console.print("[bold yellow]Context managed. Continue chatting.[/bold yellow]\n")
                    continue
                else:
                    console.print(f"\n[bold red]Generation error:[/bold red] {e}")

    except Exception as e:
        console.print(f"\n[bold red]Critical error in main loop:[/bold red] {e}")
    finally:
        console.print("\n[bold blue]Chat session ended[/bold blue]")

if __name__ == "__main__":
    main()
