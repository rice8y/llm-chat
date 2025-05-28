import os
import sys
import argparse
import re
from llama_cpp import Llama
from contextlib import contextmanager
import io
from rich import print
from rich.live import Live
from rich.spinner import Spinner as RichSpinner
from rich.console import Console
from rich.columns import Columns
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
    spinner = RichSpinner("dots")
    message_text = Text(message)
    try:
        with Live(Columns([spinner, message_text]), refresh_per_second=10, transient=True):
            result = func(*args, **kwargs)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation interrupted by user.[/bold yellow]")
        sys.exit(0)
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

    try:
        model = show_spinner("Loading model...", load_model, args)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Model loading interrupted by user.[/bold yellow]")
        sys.exit(0)

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

            try:
                spinner = RichSpinner("dots")
                reply = ""
                response = None

                with Live(Columns([spinner, Text("Thinking...")]), refresh_per_second=10, transient=True) as live:
                    response = get_response()
                    first_output = True

                    for chunk in response:
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if first_output:
                                    live.stop()
                                    first_output = False
                                console.print(content, end="", soft_wrap=True, highlight=False)
                                reply += content

                if reply:
                    print("\n")
                    messages.append({"role": "assistant", "content": reply})
                else:
                    console.print("[bold red]\n[No response generated][/bold red]")

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Generation interrupted by user.[/bold yellow]")
                continue
            except Exception as e:
                error_msg = str(e)
                if re.search(r"Requested tokens \(\d+\) exceed context window of \d+", error_msg) or \
                "could not broadcast input array from shape" in error_msg:
                    console.print("\n[bold red]Error:[/bold red] Model ran out of space in context window.")
                    choice = input("Would you like to clear chat history and continue? (y/n): ").strip().lower()
                    if choice == "y":
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."}
                        ]
                        console.print("[bold yellow]Chat history cleared. You can continue chatting.[/bold yellow]\n")
                        continue
                    else:
                        console.print("[bold yellow]Exiting chat.[/bold yellow]")
                        break
                else:
                    console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")

    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")

if __name__ == "__main__":
    main()