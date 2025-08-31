# aicli/prompts.py
# aicli: A command-line interface for interacting with AI models.
# Copyright (C) 2025 Dank A. Saurus

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


"""
A centralized collection of prompts for automated AI tasks.
"""

# --- From session_manager.py ---

HISTORY_SUMMARY_PROMPT = (
    "Concisely summarize the key facts and takeaways from the following conversation excerpt in the third person. "
    "This summary will be used as context for the rest of the conversation.\n\n"
    "--- EXCERPT ---\n{log_content}\n---"
)

MEMORY_INTEGRATION_PROMPT = (
    "You are a memory consolidation agent. Your task is to distill the crucial information from the 'NEW CHAT SESSION' "
    "and integrate it into the 'EXISTING PERSISTENT MEMORY'. Synthesize related topics, update existing facts with new "
    "information, and discard conversational fluff or trivial data. The final output must be a dense, factual summary, "
    "optimized for conciseness and relevance for a future AI to use as context. Eliminate all verbosity.\n\n"
    "--- EXISTING PERSISTENT MEMORY ---\n{existing_ltm}\n\n"
    "--- NEW CHAT SESSION TO INTEGRATE ---\n{session_content}\n\n"
    "--- UPDATED PERSISTENT MEMORY ---"
)

LOG_RENAMING_PROMPT = (
    "Based on the following chat log, generate a concise, descriptive, filename-safe title. "
    "Use snake_case. The title should be 3-5 words. "
    "Do not include any file extension like '.jsonl'. "
    "Example response: 'python_script_debugging_and_refactoring'\n\n"
    "CHAT LOG EXCERPT:\n---\n{log_content}\n---"
)

CONTINUATION_PROMPT = (
    "Please continue the conversation based on the history provided. "
    "Offer a new insight, ask a follow-up question, or rebut the last point made."
)


# --- From handlers.py ---

MULTICHAT_SYSTEM_PROMPT_OPENAI = (
    "You are OpenAI. You are not Gemini. The user is the 'Director'. Your task is to respond only as yourself. "
    "**Crucial rule: Your response must NEVER begin with `[Gemini]:` or any other participant's label.** "
    "Acknowledge and address points made by Gemini, but speak only for yourself."
)

MULTICHAT_SYSTEM_PROMPT_GEMINI = (
    "You are Gemini only. You are not OpenAI. The user is the 'Director'. Your task is to respond only as yourself. "
    "**Crucial rule: Your response must NEVER begin with `[OpenAI]:` or any other participant's label.** "
    "Acknowledge and address points made by OpenAI, but speak only for yourself."
)
