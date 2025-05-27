import os
import sys
import argparse
import time
import re
import readline
from llama_cpp import Llama
from contextlib import contextmanager
import io
from rich import print
from rich.live import Live
from rich.spinner import Spinner as RichSpinner
from rich.console import Console

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
            console.print(f"[bold red]Regex error:[/bold red] {e}")
            return
        for line in stderr_buffer:
            if not regex.search(line):
                sys.stderr.write(line)

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat with LLM.")
    parser.add_argument("--model", type=str, required=True, help="Path to the Llama model file.")
    parser.add_argument("--filename", type=str, default=None, help="Filename pattern for the model file.")
    parser.add_argument("--n_ctx", type=int, default=2048, help="Context size for the model.")
    return parser.parse_args()

def show_spinner(message, func, *args, **kwargs):
    spinner = RichSpinner("dots", text=message)
    with Live(spinner, refresh_per_second=10, transient=True):
        result = func(*args, **kwargs)
    return result

def load_model(args):
    warning_pattern = r"llama_context: n_ctx_per_seq \(\d+\) < n_ctx_train \(\d+\)"
    with suppress_specific_warning(warning_pattern):
        if os.path.exists(args.model):
            return Llama(model_path=args.model, n_ctx=args.n_ctx, verbose=False)
        elif args.filename:
            return Llama.from_pretrained(repo_id=args.model, filename=args.filename, n_ctx=args.n_ctx, verbose=False)
        else:
            raise ValueError("Model file not found and no filename pattern provided.")

def main():
    args = parse_args()

    model = show_spinner("Loading model...", load_model, args)
    console.clear()
    console.print("\n[bold green]Model loaded successfully![/bold green]")
    console.print("[dim]Enter 'exit' to quit the chat.[/dim]\n")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    try:
        while True:
            try:
                user_input = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold yellow]Exiting chat.[/bold yellow]")
                break

            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold yellow]Exiting chat.[/bold yellow]")
                break

            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            def get_response():
                return model.create_chat_completion(messages=messages.copy(), stream=True)

            response = show_spinner("Thinking...", get_response)

            reply = ""
            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        console.print(content, end="", soft_wrap=True, highlight=False)
                        reply += content

            if reply:
                print("\n")
                messages.append({"role": "assistant", "content": reply})
            else:
                console.print("[bold red]\n[No response generated][/bold red]")

    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")

if __name__ == "__main__":
    main()