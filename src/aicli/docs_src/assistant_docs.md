### AICLI Tool Documentation for AI Assistant

#### Overview
AICLI is a command-line interface for interacting with AI models like OpenAI's GPT series and Google's Gemini. It supports interactive chat, single-shot questions, multi-model conversations, and image generation.

#### Key Features
- **Personas**: Reusable configurations (`.json` files) that define an AI's behavior, system prompt, and model.
- **Context Management**: Attach local files and directories to the conversation (`-f`, `/attach`).
- **Persistent Memory**: A long-term memory file that the AI learns from across sessions.
- **Session Management**: Save and load entire conversations, preserving history and context (`/save`, `/load`, `-l`).
- **Multi-Chat**: Query OpenAI and Gemini simultaneously for comparative analysis (`-b`, `--both`).

---
### Interactive Commands (`/command`)

#### General & Session Control
- **/help**: Displays this list of commands.
- **/exit [name]**: Ends the session. Optionally provide a name for the log file.
- **/quit**: Exits immediately without updating memory or renaming the log.
- **/save [name] [--stay] [--remember]**: Saves the session to a file. Auto-generates a name if not provided.
- **/load <filename>**: Loads a session, replacing the current one.

#### AI & Model Control
- **/engine [name]**: Switches the AI engine (e.g., `openai`, `gemini`). Toggles if no name is given.
- **/model [name]**: Selects a new model for the current engine. Prompts with a list if no name is given.
- **/persona <name>**: Switches to a different persona. Use `/persona clear` to remove it.
- **/personas**: Lists all available personas.
- **/stream**: Toggles response streaming on/off.
- **/max-tokens [num]**: Sets the maximum number of tokens for the AI's response.
- **/debug**: Toggles raw API logging for the session.
- **/image [prompt]**: Initiates the interactive image generation workflow.

#### Context & Memory Management
- **/attach <path>**: Attaches a file or directory to the session context.
- **/detach <name>**: Detaches a file from the context by its filename.
- **/files**: Lists all currently attached text files.
- **/refresh [term]**: Re-reads attached files. If `[term]` is specified, only refreshes files whose names contain that term.
- **/memory**: Views the contents of the persistent memory file.
- **/remember [text]**: Injects `[text]` into persistent memory. If run without text, it consolidates the current chat into memory.

#### Information & UI
- **/clear**: Clears the current conversation history.
- **/history**: Prints the raw JSON of the current conversation history.
- **/state**: Prints the current session's configuration.
- **/set <key> <value>**: Changes an application setting (e.g., `/set stream false`).
- **/theme <name>**: Switches the display theme. Run without a name to list themes.
- **/toolbar [on|off|toggle <comp>]**: Controls the bottom toolbar.
