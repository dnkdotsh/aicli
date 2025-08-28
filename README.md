# Unified Command-Line AI Client (aicli)

A single, flexible CLI for working with OpenAI and Google Gemini:
- Chat with either engine (or both at once)
- Stream responses
- Attach files, directories, and zip archives (text and images)
- Persist and reuse memory across sessions
- Auto-summarize and trim long histories
- Generate images with OpenAI
- Keep detailed logs (raw, session, debug)

Read the important usage notes in DISCLAIMER.md before using this tool.

## Quick Start

Requirements:
- Python 3.10+ (uses modern typing like `list | None`)
- API keys:
  - OPENAI_API_KEY for OpenAI
  - GEMINI_API_KEY for Google Gemini

Install:
- macOS/Linux:
  1) git clone <repo> && cd <repo>
  2) python3 -m venv .venv && source .venv/bin/activate
  3) pip install -r requirements.txt
  4) export OPENAI_API_KEY="sk-..." (if using OpenAI)
  5) export GEMINI_API_KEY="..." (if using Gemini)
  6) python aicli.py

- Windows (PowerShell):
  1) git clone <repo>; cd <repo>
  2) py -m venv .venv; .\.venv\Scripts\Activate.ps1
  3) pip install -r requirements.txt
  4) setx OPENAI_API_KEY "sk-..." (new terminals pick this up)
  5) setx GEMINI_API_KEY "..."
  6) python aicli.py

Optional: make it executable on Unix-like systems:
- chmod +x aicli.py
- ./aicli.py

## Usage

Run without arguments to get an interactive menu:
- 1) Gemini Text (chat) [default]
- 2) OpenAI Text (chat)
- 3) OpenAI Image generation

Or specify options directly:

```
python aicli.py [-e {openai,gemini}] [-c | -i | -b PROMPT]
                [-f PATH ...] [-x PATH ...] [-m]
                [-s SESSION_NAME] [-p PROMPT]
                [--stream] [--max-tokens INT]
```

Options:
- -e, --engine: Choose engine (openai or gemini). Default is set in config.py (gemini).
- Modes (mutually exclusive):
  - -c, --chat: Chat mode (text generation).
  - -i, --image: Image generation (OpenAI only).
  - -b PROMPT, --both PROMPT: Send a single-shot prompt to both OpenAI and Gemini. Disables --engine and --session-name.
- -f PATH, --file PATH: Attach files, directories, or zip archives. Repeatable.
- -x PATH, --exclude PATH: Exclude files/directories by path or basename. Repeatable.
- -m, --memory: Load persistent memory into the system context.
- -s NAME, --session-name NAME: Custom name for chat log file (interactive chat only).
- -p PROMPT, --prompt PROMPT: Single-shot chat or image prompt.
- --stream: Enable streaming responses for chat.
- --max-tokens INT: Override model’s default max tokens for a response.

Notes:
- --image implies OpenAI engine and cannot be used with Gemini.
- --both cannot be combined with --engine or --session-name.
- When no mode is specified and you pass other args, chat mode is assumed.

## Examples

Chat (Gemini, interactive):
- python aicli.py

Chat (OpenAI, interactive):
- python aicli.py -e openai -c

Single-shot chat:
- python aicli.py -c -p "Summarize the latest Python release notes."

Streamed single-shot:
- python aicli.py -c --stream -p "Write a haiku about autumn."

Attach files and images:
- python aicli.py -c -f README.md -f docs/ -f assets.zip -p "Review these files."
- Exclude specific names or paths:
  - python aicli.py -c -f project/ -x project/node_modules -x ".DS_Store"

Use persistent memory:
- python aicli.py -c -m
- Memory is appended to the system context. At session end, a summary is integrated into persistent memory.

Custom session name:
- python aicli.py -c -s "bug_triage_session"

Ask both engines at once:
- python aicli.py -b "Explain WebSockets in simple terms." --stream

Generate an image (OpenAI only):
- python aicli.py -i -p "A watercolor painting of mountains at sunrise"
- If no -p, it will prompt (TTY) or read prompt from stdin (non-TTY).

Set max tokens:
- python aicli.py -c -p "Explain quantum teleportation." --max-tokens 500

## Interactive Chat Commands

During interactive chat, you can type:
- /exit: End the session.
- /stream: Toggle streaming on/off for this session.
- /debug: Toggle debug logging (captures raw requests/responses with sensitive data redacted).
- /clear: Clear entire conversation history (including initial context) after confirmation.
- /forget: Clear conversation history but keep initial context.
- /model: Choose a different model (lists available models from the provider).
- /engine: Switch between OpenAI and Gemini (history auto-translated to the new engine format).

## Features in Detail

- Engines and models:
  - Default engine and models are configurable in config.py.
  - On first use in a session, you will be asked whether to use the default model; you can also fetch and select from available models.
- Streaming:
  - Use --stream or toggle with /stream. Shows tokens at the end as [prompt/completion/reasoning/total] when available.
- File attachments:
  - Text files are read and embedded into the system context.
  - Images are embedded as base64 and sent with the first user message.
  - Directories and zip archives are recursively processed. Use -x to exclude.
- Persistent memory:
  - -m loads persistent memory into your context.
  - At the end of an interactive session:
    - The session is summarized.
    - That summary is integrated into persistent memory to keep it concise and up to date.
- History management:
  - Long histories are auto-summarized and trimmed mid-session once a threshold is reached (configurable in config.py).
- “Both” mode:
  - Send a single prompt to both engines and compare outputs side-by-side.
- Image generation:
  - OpenAI only. Saves generated images to images/ and logs metadata to logs/image_log.jsonl.
- Token usage:
  - After each response, token usage is shown when provided by the API.

## Files, Logs, and Directories

- config.py:
  - Defaults for engine and models.
  - Token limits and summarization thresholds.
  - Directory and log file locations.
- logs/ (created automatically):
  - raw.log: Redacted request/response logs for all API calls.
  - chat_*.jsonl or <session_name>.jsonl: Per-session chat logs.
  - debug_*.jsonl: Detailed raw logs saved when /debug is enabled.
  - persistent_memory.txt: Consolidated memory across sessions.
- images/ (created automatically):
  - Generated images saved as PNG files.
  - image_log.jsonl: Records model, prompt, and saved file path.

Sensitive data handling:
- API keys are always read from environment variables.
- Keys are redacted from logs.

## Configuration

Edit config.py to change defaults:
- DEFAULT_ENGINE: 'gemini' or 'openai'
- DEFAULT_OPENAI_CHAT_MODEL, DEFAULT_GEMINI_MODEL
- DEFAULT_OPENAI_IMAGE_MODEL
- Helper models for automatic tasks (summaries, memory integration)
- Token limits and history thresholds
- Log and image directories

## Troubleshooting

- “Error: Environment variable 'OPENAI_API_KEY'/'GEMINI_API_KEY' is not set.”
  - Ensure the corresponding environment variable is exported and available to the shell running aicli.
- HTTP Request Error or API Error:
  - Check model names and account access.
  - Verify your API key and provider status.
  - See logs/raw.log and, if enabled, debug_*.jsonl for redacted diagnostics.
- Streaming connection interrupted:
  - The client prints a warning and returns the text received so far.
- Model list fetch failures:
  - The client will fall back to the default model if fetching available models fails.

## Notes and Limitations

- Image generation is available only with OpenAI.
- --both mode is single-shot only and cannot be used with --engine or --session-name.
- When no mode is provided and you pass other args, the app defaults to chat mode.
- For single-shot chat, use -p/--prompt. Interactive chat reads your input live; chat single-shot does not read from stdin.
- See DISCLAIMER.md for important information about costs, reliability, and liability.

## Contributing

Issues and pull requests are welcome. Please include steps to reproduce and relevant logs (with any additional sensitive info removed).

