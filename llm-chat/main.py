import os
import sys
import argparse
import threading
import time
import re
from llama_cpp import Llama
from contextlib import contextmanager
import io

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
        for line in stderr_buffer:
            if not re.search(pattern, line):
                sys.stderr.write(line)

class Spinner:
    def __init__(self, message="Loading"):
        self.message = message
        self.spinner = ['▖', '▘', '▝', '▗']
        self.index = 0
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def _spin(self):
        while self.running:
            print(f"\r{self.message} {self.spinner[self.index % len(self.spinner)]}", end='', flush=True)
            self.index += 1
            time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        spinner_length = len(self.message) + 3
        print(f"\r{' ' * spinner_length}\r", end='', flush=True)
        time.sleep(0.05)

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat with Llama model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the Llama model file.")
    parser.add_argument("--filename", type=str, default=None, help="Filename pattern for the model file.")
    return parser.parse_args()

def main():
    args = parse_args()

    spinner = Spinner("Loading model")
    spinner.start()

    warning_pattern = r"llama_context: n_ctx_per_seq \(\d+\) < n_ctx_train \(\d+\)"

    with suppress_specific_warning(warning_pattern):
        if os.path.exists(args.model):
            model = Llama(model_path=args.model, verbose=False)
        else:
            if not args.filename:
                raise ValueError("Model file not found and no filename pattern provided.")
            model = Llama.from_pretrained(repo_id=args.model, filename=args.filename, verbose=False)

    spinner.stop()

    print("Enter 'exit' to quit the chat.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    conversation_count = 0

    try:
        while True:
            try:
                user_input = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break

            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})
            conversation_count += 1

            spinner = Spinner("Thinking")
            spinner.start()

            try:
                api_messages = messages.copy()
                response = model.create_chat_completion(messages=api_messages, stream=True)

                reply = ""
                spinner_stopped = False
                first_chunk = True

                for chunk in response:
                    if not spinner_stopped:
                        spinner.stop()
                        spinner_stopped = True
                        if first_chunk:
                            first_chunk = False

                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            reply += content

                if reply:
                    print("\n")
                    messages.append({"role": "assistant", "content": reply})
                else:
                    print("[No response generated]")

            except Exception as e:
                if not spinner_stopped:
                    spinner.stop()
                print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()