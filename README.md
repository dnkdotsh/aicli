# AICLI: A Unified Command-Line AI Client

A flexible and robust command-line interface for interacting with multiple AI models, including OpenAI (GPT series) and Google (Gemini series).

- **Multi-Engine Chat**: Engage with OpenAI or Gemini, or have them respond to the same prompt simultaneously for comparative analysis.
- **Reusable Personas**: Define and switch between different AI "personas" with pre-configured system prompts, models, and settings for different tasks.
- **Dynamic Context Control**: Attach, detach, list, and refresh local files, directories, or even zip archives mid-conversation to provide deep, evolving context.
- **Powerful Memory System**: Leverage a long-term memory file that the AI learns from across sessions. View, inject facts into, or consolidate conversations into memory on the fly.
- **Full Session Management**: Save your interactive sessions to a file and resume them later, preserving the full conversation history, model state, and context.
- **Image Generation**: Generate images using DALL-E 3 directly from the command line.
- **Smart & Configurable**: Warns about potentially expensive large contexts, automatically manages conversation history to stay within token limits, and allows for deep customization via a simple settings file.

---

## Installation

### Requirements
-   Python 3.10+
-   `pipx` (Recommended for CLI tools. Install with `pip install pipx`)
-   API keys for [OpenAI](https://platform.openai.com/api-keys) and/or [Google AI](https://aistudio.google.com/app/apikey)

### User Installation (Recommended)
`pipx` installs Python applications in isolated environments, making them available globally without interfering with other Python projects.

From the project's root directory:
```bash
# Install the aicli command globally
pipx install .
```

### Developer Installation
If you plan to modify the code, install it in editable mode within a virtual environment.

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with test dependencies
pip install -e ".[test]"
```

---

## Configuration

### 1. API Keys
Create a `.env` file in the project root by copying the example:

```bash
cp .env.example .env
```
Edit the new `.env` file and add your secret API keys. The application will load them automatically at startup.

### 2. Application Settings
User-configurable settings (like default models and token limits) are stored in `settings.json`.

-   **Location:** `~/.config/aicli/settings.json`
-   This file is created automatically on the first run. You can edit it directly or manage settings from within an interactive session using `/set`.

### 3. Personas
A "persona" is a reusable configuration saved as a JSON file. It allows you to pre-define the system prompt, AI model, and other settings for specific tasks.

-   **Location:** `~/.config/aicli/personas/`
-   This directory is created on the first run with a default `aicli_assistant.json` persona. You can create as many `.json` files as you need.

An example `code_reviewer.json` file might look like this:
```json
{
  "name": "Code Reviewer",
  "description": "A meticulous code reviewer that focuses on bugs and best practices.",
  "engine": "openai",
  "model": "gpt-4o",
  "system_prompt": "You are an expert code reviewer. Analyze the provided code for logical errors, security vulnerabilities, and deviations from best practices. Provide your feedback in a structured list, citing specific line numbers. Be concise and direct."
}
```

---

## Usage

The application has two main commands: `chat` (the default) and `review`.

```
aicli [chat|review] [OPTIONS]
```

### `chat` (Default Command)
If no command is specified, `chat` is assumed.

```
aicli [-e {openai,gemini}] [-m MODEL] [--system-prompt PROMPT_OR_PATH]
      [-P PERSONA]
      [-c | -i | -b [PROMPT] | -l FILEPATH]
      [-p PROMPT] [-f PATH] [-x PATH] [--memory]
      [-s NAME] [--stream | --no-stream] [--max-tokens INT] [--debug]
```

#### Modes (mutually exclusive)
-   `-c, --chat`: Interactive chat mode (default).
-   `-i, --image`: Image generation mode (OpenAI only).
-   `-b, --both [PROMPT]`: Multi-chat mode to query both OpenAI and Gemini simultaneously.
-   `-l, --load FILEPATH`: Load a saved chat session and resume interaction.

#### Context & Input Arguments
-   `-p, --prompt PROMPT`: A single, non-interactive prompt.
-   `-P, --persona PERSONA`: Start the session with a specific persona.
-   `--system-prompt PROMPT_OR_PATH`: Provide a system instruction as a string or a path to a text file. Overrides a persona's system prompt.
-   `-f, --file PATH`: Attach a file or directory. Can be used multiple times.
-   `-x, --exclude PATH`: Exclude a file or directory from being processed.
-   `--memory`: Toggles the use of persistent memory for the session (reverses the default in `settings.json`).

### `review` Command
Launch an interactive TUI to browse, replay, rename, or delete past chat logs and saved sessions.

```bash
# Launch the interactive review tool
aicli review

# Directly replay a specific session file
aicli review ~/.local/share/aicli/sessions/my_session.json
```

---

## Examples

**Start an interactive chat with the default engine:**
```bash
aicli
```

**Start a session with a specific persona:**
```bash
aicli -P code_reviewer -f ./src/
```

**Ask a single-shot question without entering interactive mode:**
```bash
aicli -p "What is the capital of Nebraska?"
```

**Analyze code by attaching a directory and providing a system prompt:**
```bash
aicli --system-prompt "You are a senior Python developer. Review this code for bugs." -f ./src/
```

**Ask both models the same question for comparison:**
```bash
aicli --both "Compare and contrast Python's asyncio with traditional threading."
```

**Generate an image (OpenAI only):**
```bash
aicli -i -p "A photorealistic image of a red panda programming on a laptop"
```

**Have a dynamic interactive session:**
```bash
# Start an interactive session with the default persona
aicli -P aicli_assistant
You: How do I save a session?
...
# Switch personas mid-conversation
You: /persona code_reviewer
You: /attach ./src/main.py
You: Review this code for me.
...
# Exit the session, giving the log a specific name
You: /exit main_py_review
```

---

## Interactive Chat Commands

During an interactive chat session, type these commands instead of a prompt:

#### **General Commands**
-   `/exit [name]`: End the current session. Optionally provide a name for the log file.
-   `/quit`: Exit immediately without saving memory or renaming the log.
-   `/help`: Display this help message.
-   `/clear`: Clear the current conversation history.
-   `/history`: Print the raw JSON of the current conversation history.
-   `/state`: Print the current session's configuration (model, engine, attached files, etc.).

#### **Session & Context Management**
-   `/save [name] [--stay]`: Save the session to a file. Auto-generates a name if not provided. Use `--stay` to continue the session after saving.
-   `/load <filename>`: Load a session, replacing the current one.
-   `/refresh [name]`: Re-read attached files. If `[name]` is provided, only refreshes files whose names contain the text.
-   `/attach <path>`: Attach a file to the session context.
-   `/detach <name>`: Detach a file from the context by its filename.
-   `/files`: List all currently attached text files, sorted by size.

#### **Memory Management**
-   `/memory`: View the contents of the persistent memory file.
-   `/remember [text]`: Injects a specific `[text]` into persistent memory. If run without text, it consolidates the current conversation into memory.

#### **AI & Model Control**
-   `/engine [name]`: Switch between `openai` and `gemini`. The history is automatically translated.
-   `/model [name]`: Change the model for the current engine.
-   `/persona <name>`: Switch to a different persona. Use `/persona clear` to remove it.
-   `/personas`: List all available personas from your configuration directory.
-   `/stream`: Toggle response streaming on or off for the current session.
-   `/max-tokens <num>`: Set the max output tokens for the current session.
-   `/debug`: Toggle raw API logging for the current session.
-   `/set [key] [value]`: Change a default setting (e.g., `/set default_max_tokens 8192`). Run without arguments to list all settings.

---

## File & Directory Layout

-   `src/aicli/`: Contains all application source code.
-   `tools/`: Contains utility scripts for packaging and development.
-   `~/.config/aicli/`: Stores the user `settings.json` file and the `personas/` directory.
-   `~/.local/share/aicli/`: Stores all application data:
    -   `logs/`: Contains session chats, raw API calls, and debug logs.
    -   `images/`: Where generated images are saved.
    -   `sessions/`: Where named sessions are saved with `/save`.
    -   `persistent_memory.txt`: The long-term memory file for the AI.

---

## Final thoughts

<small>I literally just made this to make the AI argue against each other. I hope someone else can find a better use for it.</small>
