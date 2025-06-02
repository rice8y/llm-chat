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


def calculate_optimal_batch_size(n_ctx: int, n_gpu_layers: int) -> int:
    base_batch = min(512, n_ctx // 4)
    if n_gpu_layers > 0:
        return min(base_batch * 2, 2048)
    return base_batch


@contextmanager
def suppress_stderr_warnings(regex_pattern: str):
    stderr_backup = sys.stderr
    stderr_buffer = io.StringIO()
    sys.stderr = stderr_buffer
    
    try:
        yield
    finally:
        sys.stderr = stderr_backup
        regex = re.compile(regex_pattern)
        
        if not stderr_buffer.getvalue():
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
        else:
            return recent_messages
    return messages


def show_spinner(message: str, func, *args, **kwargs):
    spinner = RichSpinner("dots", text=f"[cyan]{message}[/cyan]")
    message_text = Text(message, style="cyan")
    live_display = Columns([spinner, message_text], expand=False)
    try:
        with Live(live_display, refresh_per_second=10, transient=True, console=console):
            result = func(*args, **kwargs)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Interrupted: {message}[/yellow]")
        raise
    return result


def parse_arguments():
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
                        help="Enable mul_mat_q optimization")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--n_keep", type=int, default=0,
                        help="Number of tokens to keep from initial prompt")
    parser.add_argument("--use_mmap", action="store_true", default=True,
                        help="Use memory mapping for model loading")
    parser.add_argument("--profile", action="store_true",
                        help="Enable performance profiling")

    return parser.parse_args()


def load_model(args):
    warning_pattern_n_ctx = r"llama_context: n_ctx_per_seq \(\d+\) < n_ctx_train \(\d+\)"

    if args.n_batch is None:
        args.n_batch = calculate_optimal_batch_size(args.n_ctx, args.n_gpu_layers)
        console.print(f"[dim]Auto-calculated batch size: {args.n_batch}[/dim]")

    llama_cpp_kwargs = {
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "n_threads": args.n_threads,
        "n_batch": args.n_batch,
        "use_mmap": args.use_mmap,
        "verbose": False,
        "n_keep": args.n_keep,
    }

    if args.flash_attn:
        llama_cpp_kwargs["flash_attn"] = True
        console.print("[dim]Flash Attention enabled[/dim]")

    if args.mul_mat_q:
        llama_cpp_kwargs["mul_mat_q"] = True
        console.print("[dim]mul_mat_q optimization enabled[/dim]")

    with suppress_stderr_warnings(warning_pattern_n_ctx):
        if os.path.isfile(args.model):
            return Llama(model_path=args.model, **llama_cpp_kwargs)
        elif args.filename:
            return Llama.from_pretrained(
                repo_id=args.model,
                filename=args.filename,
                **llama_cpp_kwargs
            )
        else:
            raise ValueError("Invalid model path or missing filename for Hugging Face model")


async def get_llm_response_stream(model, messages, temperature, top_p):
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


def get_user_input(prompt: str, gpu_enabled: bool) -> str:
    try:
        if gpu_enabled:
            return console.input(prompt)
        else:
            return console.input(prompt)
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt from None


def handle_context_overflow(messages: list[dict[str, str]], args) -> list[dict[str, str]]:
    console.print(f"\n[bold red]Context limit reached ({args.n_ctx} tokens)[/bold red]")
    console.print("[yellow]Options:[/yellow]")
    console.print("1. Clear conversation history (c)")
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
    elif choice == "e":
        return None
    else:
        console.print("[red]Invalid choice. Reducing context automatically.[/red]")
        return manage_context_efficiently(messages, args.n_ctx)


def main():
    args = parse_arguments()
    
    try:
        model = show_spinner(
            "Loading model...",
            load_model,
            args
        )
    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        return

    console.clear()
    console.print("\n[bold green]Model loaded successfully![/bold green]")

    config_info = [
        f"Context: {args.n_ctx}",
        f"GPU Layers: {args.n_gpu_layers}",
        f"Threads: {args.n_threads}",
        f"Batch: {args.n_batch}",
        f"Temperature: {args.temperature}",
        f"Top-p: {args.top_p}",
        f"Memory Map: {args.use_mmap}"
    ]

    rprint(f"[dim]{' | '.join(config_info)}[/dim]")
    console.print("[dim]Enter 'exit', 'quit', or 'clear' to manage chat[/dim]\n")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        try:
            user_input = get_user_input("[bold blue]You:[/bold blue] ", args.n_gpu_layers != 0)
            
            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold yellow]Exiting chat[/bold yellow]")
                break

            if user_input.lower() == "clear":
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                console.print("[bold green]Conversation cleared[/bold green]\n")
                continue

            messages.append({"role": "user", "content": user_input})

            if estimate_tokens(messages) > args.n_ctx * 0.9:
                messages = handle_context_overflow(messages, args)
                if messages is None:
                    break

            console.print("\n[bold green]Assistant:[/bold green] ", end="")
            
            reply_content = ""
            start_time = time.time()
            first_chunk_received = False
            token_count = 0

            spinner = RichSpinner("dots", text="[dim]Thinking...[/dim]")
            with Live(spinner, refresh_per_second=10, transient=True, console=console) as live:
                current_messages = messages.copy()  # Capture messages at this point
                
                def get_response():
                    return model.create_chat_completion(
                        messages=current_messages,
                        stream=True,
                        temperature=args.temperature,
                        top_p=args.top_p
                    )

                response_stream = get_response()
                
                for chunk in response_stream:
                    if not first_chunk_received:
                        live.stop()
                        first_chunk_received = True

                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    
                    if "content" in delta:
                        content_piece = delta["content"]
                        token_count += 1
                        console.print(content_piece, end="", soft_wrap=True, highlight=False)
                        reply_content += content_piece

                    if choice.get("finish_reason") is not None:
                        break

            generation_time = time.time() - start_time
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0
            console.print(f"\n[dim]Generation: {generation_time:.3f}s | Tokens: {token_count} | Speed: {tokens_per_second:.1f} tok/s[/dim]")

            if reply_content:
                console.print()
                messages.append({"role": "assistant", "content": reply_content})

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error: {e}[/bold red]")
            continue


if __name__ == "__main__":
    main()