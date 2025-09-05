# AICLI: A Unified Command-Line AI Client

A flexible and robust command-line interface for interacting with multiple AI models, including OpenAI (GPT series) and Google (Gemini series).

## Key Features

-   **Multi-Engine Chat**: Engage with OpenAI or Gemini, or have them respond to the same prompt simultaneously for comparative analysis.
-   **Reusable Personas**: Define and switch between different AI "personas" with pre-configured system prompts, models, and settings for different tasks.
-   **Dynamic Context Management**: Attach, detach, list, and refresh local files, directories, or even zip/tar archives mid-conversation to provide deep, evolving context.
-   **Powerful Memory System**: Leverage a long-term memory file that the AI learns from across sessions. View, inject facts into, or consolidate conversations into memory on the fly.
-   **Full Session Management**: Save your interactive sessions to a file and resume them later, preserving the full conversation history, model state, and context.
-   **Image Generation**: Generate images using DALL-E 3 directly from the command line or within an interactive chat.
-   **Highly Configurable UI**: Customize prompt colors and the content of the live information toolbar directly from the command line.
-   **Smart & Cross-Platform**: Warns about potentially expensive large contexts, automatically manages conversation history to stay within token limits, and stores configuration/data in OS-native locations (e.g., `~/.config/aicli` on Linux).

---

## Installation

### Requirements
-   Python 3.10+
-   API keys for [OpenAI](https://platform.openai.com/api-keys) and/or [Google AI](https://aistudio.google.com/app/apikey)

The application is fully cross-platform and will run anywhere Python is supported.

### Recommended (via pipx)
`pipx` installs Python applications in isolated environments, making them available globally without interfering with other Python projects.

```bash
# Install pipx if you don't have it
pip install pipx

# Install from the project's root directory
pipx install .
```

### Alternate (via pip)
You can also install directly with `pip`.

```bash
# Install from the project's root directory
pip install .
```

### Developer Installation
If you plan to modify the code, install it in editable mode within a virtual environment.

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with dev/test dependencies
pip install -e ".[dev,test]"
```
_Note: A bash-based easy-install script is planned for a future release. (Although how much easier does it get?)_

---

## Configuration

On the first run, `aicli` will guide you through creating the necessary directories and a `.env` file for your API keys.

### 1. Application Settings & Personas
User-configurable files are stored in your OS-native config directory (e.g., `~/.config/aicli/` on Linux).

-   **Settings:** `settings.json` contains defaults for models, colors, and behavior. You can edit this file directly or use the `/set`, `/style`, and `/toolbar` commands in an interactive session.
-   **Personas:** The `personas/` directory holds reusable configurations (JSON files) that define an AI's system prompt, model, and other settings. A default `aicli_assistant.json` is created for you.

An example `code_reviewer.json` persona:
```json
{
  "name": "Code Reviewer",
  "description": "A meticulous code reviewer that focuses on bugs and best practices.",
  "engine": "openai",
  "model": "gpt-4o",
  "system_prompt": "You are an expert code reviewer. Analyze the provided code for logical errors, security vulnerabilities, and deviations from best practices. Provide your feedback in a structured list, citing specific line numbers. Be concise and direct."
}
```
### 2. Application Data
Generated data is stored in your OS-native data directory (e.g., `~/.local/share/aicli/` on Linux). This includes `logs/`, saved `sessions/`, `images/`, and the `persistent_memory.txt` file.

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
-   `-x, --exclude PATH`: Exclude a file or directory from being processed by `-f`.
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

**Start a session with a specific persona and attach a source code directory:**
```bash
aicli -P code_reviewer -f ./src/
```

**Ask a single-shot question piped from another command:**
```bash
echo "What is the capital of Nebraska?" | aicli
```

**Generate an image (OpenAI only):**
```bash
aicli -i -p "A photorealistic image of a red panda programming on a laptop"
```

**Ask both models the same question for comparison:**
```bash
aicli --both "Compare and contrast Python's asyncio with traditional threading."
```

---

## Interactive Commands

During an interactive session, type these commands instead of a prompt.

### General Commands
-   `/help`: Display this help message.
-   `/exit [name]`: End the session. Optionally provide a name for the log file.
-   `/quit`: Exit immediately without updating memory or renaming the log.
-   `/clear`: Clear the current conversation history.
-   `/history`: Print the raw JSON of the current conversation history.
-   `/state`: Print the current session's configuration.

### Context, Session & Memory
-   `/save <name> [--stay] [--remember]`: Save the session. Auto-generates a name if not provided.
    -   `--stay`: Do not exit after saving.
    -   `--remember`: Update persistent memory with the session content.
-   `/load <filename>`: Load a session, replacing the current one.
-   `/attach <path>`: Attach a file or directory to the session context.
-   `/detach <name>`: Detach a file from the context by its filename.
-   `/files`: List all currently attached text files, sorted by size.
-   `/refresh [term]`: Re-read attached files. If `[term]` is given, only refreshes files whose names contain that term.
-   `/memory`: View the contents of the persistent memory file.
-   `/remember [text]`: Injects `[text]` into persistent memory. If run without text, it consolidates the current chat into memory.

### AI & Model Control
-   `/engine [name]`: Switch AI engine (`openai` or `gemini`). Toggles if no name is given.
-   `/model [name]`: Select a new model for the current engine. Prompts with a list if no name is given.
-   `/persona <name>`: Switch to a different persona. Use `/persona clear` to remove it.
-   `/personas`: List all available personas.
-   `/stream`: Toggle response streaming on/off.
-   `/max-tokens <num>`: Set the max output tokens for the session.
-   `/debug`: Toggle raw API logging for the session.
-   `/image [prompt]`: Initiate the interactive image generation workflow.

### UI & Settings
-   `/set <key> <value>`: Change an application setting (e.g., `/set stream false`).
-   `/toolbar [on|off|toggle <comp>]`: Control the bottom toolbar. Components: `io`, `live`, `model`, `persona`.
-   `/style <prompt|toolbar> <comp> <val>`: Change display colors (e.g., `/style toolbar model 'fg:ansired bold'`).

### Multi-Chat Commands
-   `/ai <gpt|gem> [prompt]`: Send a targeted prompt to only one AI. If no prompt is given, the AI is asked to continue.
-   All general commands like `/exit`, `/quit`, `/help`, `/history`, `/clear`, `/debug`, and `/save` are also available.

---

### Final Thoughts
<small>I literally just made this to make the AI argue against each other. I hope someone else can find a better use for it.</small>
