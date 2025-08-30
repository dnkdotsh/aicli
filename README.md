# Unified Command-Line AI Client (aicli)

A flexible and robust command-line interface for interacting with OpenAI and Google Gemini models.

-   Chat with either engine, or get simultaneous responses from both.
-   Stream responses for immediate feedback.
-   Attach local files, directories, and zip archives to provide context.
-   Leverage a persistent memory system that learns across sessions.
-   Automatically summarize and trim long conversation histories to manage context.
-   Generate images using DALL-E.
-   Configure behavior with a simple settings file and `.env` for API keys.

---
## Installation

This tool is designed as a command-line application. The recommended way to install it is using `pipx`, which installs Python applications in isolated environments.

### ## Requirements
-   Python 3.10+
-   `pipx` (Install with `pip install pipx`)
-   API keys for [OpenAI](https://platform.openai.com/api-keys) and/or [Google AI](https://aistudio.google.com/app/apikey)

### ## User Installation (Recommended)
From the project root directory:
```bash
# Install the application using pipx
pipx install .
```

This will install the `aicli` command globally and make it available in your shell.

### \#\# Developer Installation

If you plan to modify the code, install it in editable mode within a virtual environment.

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with test dependencies
pip install -e ".[test]"
```

-----

## Configuration

### \#\# 1. API Keys

Create a file named `.env` in the project root by copying the example:

```bash
cp .env.example .env
```

Now, edit the `.env` file and add your secret API keys. The application will automatically load them at startup.

### \#\# 2. Application Settings

User-configurable settings (like default models and token limits) are stored in `settings.json`.

  - **Location:** `~/.config/aicli/settings.json`
  - This file is created automatically the first time you run the app or use the `/set` command. You can edit it directly or manage settings from within an interactive session.

-----

## Usage

Once installed, use the `aicli` command.

```
aicli [-e {openai,gemini}] [-m MODEL] [--system-prompt PROMPT_OR_PATH]
      [-c | -i | -b [PROMPT]]
      [-p PROMPT] [-f PATH] [-x PATH] [--memory]
      [-s NAME] [--stream | --no-stream] [--max-tokens INT] [--debug]
```

### \#\# Modes (mutually exclusive)

  - `-c, --chat`: Interactive chat mode. This is the default if other arguments imply it.
  - `-i, --image`: Image generation mode (OpenAI only).
  - `-b, --both [PROMPT]`: Multi-chat mode to query both OpenAI and Gemini simultaneously.

### \#\# Context Arguments

  - `-p, --prompt PROMPT`: A single, non-interactive prompt.
  - `--system-prompt PROMPT_OR_PATH`: Provide a system instruction as a string or a path to a text file.
  - `-f, --file PATH`: Attach a file or directory. Can be used multiple times.
  - `--memory`: Force the use of persistent memory for the session.

-----

## Examples

**Start an interactive chat with the default engine (Gemini):**

```bash
aicli
```

**Start an interactive chat with OpenAI:**

```bash
aicli -e openai
```

**A single-shot, non-interactive question:**

```bash
aicli -p "What is the capital of Nebraska?"
```

**Analyze code by attaching a directory and providing a system prompt:**

```bash
aicli --system-prompt "You are a senior Python developer. Review this code for bugs." -f ./src/
```

**Ask both models the same question:**

```bash
aicli --both "Compare and contrast Python's asyncio with traditional threading."
```

**Generate an image (OpenAI only):**

```bash
aicli -i -p "A photorealistic image of a red panda programming on a laptop"
```

-----

## Interactive Chat Commands

During an interactive chat session, type these commands instead of a prompt:

  - `/exit`: End the current session.
  - `/help`: Display this help message.
  - `/stream`: Toggle response streaming on or off.
  - `/debug`: Toggle raw API logging for the current session.
  - `/memory`: Toggle whether the session will be saved to persistent memory.
  - `/clear`: Clear the current conversation history.
  - `/history`: Print the raw JSON of the current conversation history.
  - `/state`: Print the current session's configuration (model, engine, etc.).
  - `/engine [name]`: Switch between `openai` and `gemini`. The history is automatically translated.
  - `/model [name]`: Change the model for the current engine.
  - `/set <key> <value>`: Change a default setting (e.g., `/set default_max_tokens 8192`).
  - `/max-tokens <num>`: Set the max output tokens for the current session.

-----

## File & Directory Layout

  - `src/aicli/`: Contains all the application source code.
  - `tools/`: Contains utility scripts for packaging and development (`package.py`, `sync.py`).
  - `~/.config/aicli/`: Stores the user `settings.json` file.
  - `~/.local/share/aicli/`: Stores all application data, including:
      - `logs/`: Contains session chats, raw API calls, and debug logs.
      - `images/`: Where generated images are saved.
      - `persistent_memory.txt`: The long-term memory file for the AI.

<!-- end list -->

## Final thoughts

<small>I literally just made this to make the AI argue against each other. I hope someone else can find a better use for it.</small>
