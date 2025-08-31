# AICLI: A Unified Command-Line AI Client

A flexible and robust command-line interface for interacting with multiple AI models, including OpenAI (GPT series) and Google (Gemini series).

- **Multi-Engine Chat**: Engage with OpenAI or Gemini, or have them respond to the same prompt simultaneously for comparative analysis.
- **Rich Context**: Attach local files, directories, or even zip archives to provide deep context for your conversations.
- **Persistent Memory**: Leverage a long-term memory system that allows the AI to learn and retain information across different chat sessions.
- **Session Management**: Save your interactive sessions to a file and resume them later, preserving the full conversation history and context.
- **Live Context Refresh**: Update the content of attached files mid-session with the `/refresh` command, perfect for iterative coding and analysis.
- **Image Generation**: Generate images using DALL-E 3 directly from the command line.
- **Smart & Configurable**: Automatically manages conversation history to stay within token limits and allows for deep customization via a simple settings file.

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
-   This file is created automatically on the first run or when using the `/set` command. You can edit it directly or manage settings from within an interactive session.

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
-   `--system-prompt PROMPT_OR_PATH`: Provide a system instruction as a string or a path to a text file.
-   `-f, --file PATH`: Attach a file or directory. Can be used multiple times.
-   `-x, --exclude PATH`: Exclude a file or directory from being processed.
-   `--memory`: Toggles the use of persistent memory for the session (reverses the default in `settings.json`).

### `review` Command
Launch an interactive TUI to browse, replay, rename, or delete past chat logs and saved sessions.

```
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

**Save a session, then resume it later:**
```bash
# Start an interactive session and attach a project
aicli -f ./my-project/

# ... have a conversation ...
# Then, inside the session, save it and exit
You: /save my_project_review

# Later, resume the session from the command line
aicli --load my_project_review.json
```

---

## Interactive Chat Commands

During an interactive chat session, type these commands instead of a prompt:

-   `/exit`: End the current session.
-   `/help`: Display this help message.
-   `/clear`: Clear the current conversation history.
-   `/history`: Print the raw JSON of the current conversation history.
-   `/state`: Print the current session's configuration (model, engine, attached files, etc.).

**Session & Context Management:**
-   `/save [name] [--stay]`: Save the session to a file. Auto-generates a name if not provided. Use `--stay` to continue the session after saving.
-   `/load <filename>`: Load a session, replacing the current one.
-   `/refresh [name]`: Re-read attached files. If `[name]` is provided, only refreshes files whose names contain the text.
-   `/memory`: Toggle whether the session will be saved to persistent memory on exit.

**AI & Model Control:**
-   `/engine [name]`: Switch between `openai` and `gemini`. The history is automatically translated.
-   `/model [name]`: Change the model for the current engine.
-   `/stream`: Toggle response streaming on or off for the current session.
-   `/max-tokens <num>`: Set the max output tokens for the current session.
-   `/debug`: Toggle raw API logging for the current session.
-   `/set <key> <value>`: Change a default setting (e.g., `/set default_max_tokens 8192`).

---

## File & Directory Layout

-   `src/aicli/`: Contains all application source code.
-   `tools/`: Contains utility scripts for packaging and development.
-   `~/.config/aicli/`: Stores the user `settings.json` file.
-   `~/.local/share/aicli/`: Stores all application data:
    -   `logs/`: Contains session chats, raw API calls, and debug logs.
    -   `images/`: Where generated images are saved.
    -   `sessions/`: Where named sessions are saved with `/save`.
    -   `persistent_memory.txt`: The long-term memory file for the AI.

---

## Final thoughts

<small>I literally just made this to make the AI argue against each other. I hope someone else can find a better use for it.</small>
