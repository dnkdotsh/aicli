# tests/test_session_manager_main_loop.py
"""
Tests for the core interactive loops and helper functions in aicli/session_manager.py.
"""

import json
import logging
from pathlib import Path

import pytest
from aicli import config, utils
from aicli.session_manager import (
    SessionState,
    _assemble_full_system_prompt,
    _condense_chat_history,
    perform_interactive_chat,
)


@pytest.fixture
def session_state(mock_openai_engine):
    """Provides a fresh, default SessionState for each test."""
    return SessionState(
        engine=mock_openai_engine,
        model="gpt-4o-mini",
        system_prompt="Base prompt.",
        initial_system_prompt=None,
        current_persona=None,
        max_tokens=4096,
        memory_enabled=True,
    )


def test_assemble_full_system_prompt(session_state):
    """Tests that the system prompt and attachments are correctly combined."""
    # Test with only a system prompt
    assert _assemble_full_system_prompt(session_state) == "Base prompt."

    # Test with attachments
    session_state.attachments = {
        Path("/test/file1.py"): "content1",
        Path("/test/file2.md"): "content2",
    }
    full_prompt = _assemble_full_system_prompt(session_state)
    assert full_prompt.startswith("Base prompt.")
    assert "--- ATTACHED FILES ---" in full_prompt
    assert "--- FILE: /test/file1.py ---\ncontent1" in full_prompt
    assert "--- FILE: /test/file2.md ---\ncontent2" in full_prompt

    # Test with no system prompt but with attachments
    session_state.system_prompt = None
    full_prompt_no_base = _assemble_full_system_prompt(session_state)
    assert not full_prompt_no_base.startswith("Base prompt.")
    assert "--- ATTACHED FILES ---" in full_prompt_no_base


def test_condense_chat_history_success(session_state, mocker):
    """Tests that chat history is correctly summarized and replaced on success."""
    long_history = []
    for i in range(config.HISTORY_SUMMARY_THRESHOLD_TURNS * 2):
        long_history.append(utils.construct_user_message("openai", f"User {i}", []))
        long_history.append(
            utils.construct_assistant_message("openai", f"Assistant {i}")
        )
    session_state.history = long_history

    mock_api_call = mocker.patch(
        "aicli.api_client.perform_chat_request",
        return_value=("This is the summary.", {}),
    )

    _condense_chat_history(session_state)

    mock_api_call.assert_called_once()
    num_messages_to_trim = config.HISTORY_SUMMARY_TRIM_TURNS * 2
    expected_len = (len(long_history) - num_messages_to_trim) + 1
    assert len(session_state.history) == expected_len
    summary_message = session_state.history[0]
    summary_text = utils.extract_text_from_message(summary_message)
    assert "[PREVIOUSLY DISCUSSED]:" in summary_text
    assert "This is the summary." in summary_text


def test_condense_chat_history_api_failure(session_state, mocker, caplog):
    """Tests that history is not condensed if the summarization API call fails."""
    long_history = [
        utils.construct_user_message("openai", "User", []) for _ in range(24)
    ]
    session_state.history = long_history.copy()

    mocker.patch(
        "aicli.api_client.perform_chat_request",
        return_value=("API Error: Failed", {}),
    )

    with caplog.at_level(logging.WARNING):
        _condense_chat_history(session_state)
        assert "History summarization failed" in caplog.text

    assert session_state.history == long_history


def test_perform_interactive_chat_large_attachment_warning_proceed(
    session_state, fake_fs, mocker
):
    """Tests that the large attachment warning appears and the session continues on 'yes'."""
    large_file_path = Path("/fake/large_file.txt")
    # Use pyfakefs to create the file with the desired size
    fake_fs.create_file(
        large_file_path,
        contents=" " * (config.LARGE_ATTACHMENT_THRESHOLD_BYTES + 1),
    )
    session_state.attachments = {large_file_path: "content"}

    mock_prompt = mocker.patch(
        "aicli.session_manager.prompt", side_effect=["yes", EOFError]
    )

    perform_interactive_chat(session_state, "test_session")
    # The argument is an ANSI object for color, so we must cast it to a string for the assertion
    assert "WARNING: Total size of attached files is" in str(
        mock_prompt.call_args_list[0][0][0]
    )


def test_perform_interactive_chat_large_attachment_warning_abort(
    session_state, fake_fs, mocker
):
    """Tests that the session aborts if the user does not confirm the large attachment warning."""
    large_file_path = Path("/fake/large_file.txt")
    # Use pyfakefs to create the file with the desired size
    fake_fs.create_file(
        large_file_path,
        contents=" " * (config.LARGE_ATTACHMENT_THRESHOLD_BYTES + 1),
    )
    session_state.attachments = {large_file_path: "content"}

    mock_prompt = mocker.patch(
        "aicli.session_manager.prompt", side_effect=["no", EOFError]
    )

    perform_interactive_chat(session_state, "test_session")
    assert mock_prompt.call_count == 1  # Only the confirmation prompt should be called


def test_perform_interactive_chat_empty_input_is_ignored(session_state, mocker):
    """Tests that empty or whitespace-only input does not trigger an API call."""
    mock_api_call = mocker.patch("aicli.api_client.perform_chat_request")

    perform_interactive_chat(session_state, "test_session")
    mock_api_call.assert_not_called()
    assert len(session_state.history) == 0


def test_perform_interactive_chat_finalization_custom_rename(
    session_state, fake_fs, mocker
):
    """Tests that the log is correctly renamed when a custom name is provided via state."""
    session_state.custom_log_rename = "my_custom_name"
    # Simulate one turn so the log file is created
    mocker.patch("aicli.session_manager.prompt", side_effect=["a prompt", EOFError])
    mocker.patch(
        "aicli.api_client.perform_chat_request", return_value=("an answer", {})
    )
    mocker.patch("aicli.commands.consolidate_session_into_memory")
    mock_auto_rename = mocker.patch("aicli.commands.rename_session_log")

    perform_interactive_chat(session_state, "initial_name")

    assert not (config.LOG_DIRECTORY / "initial_name.jsonl").exists()
    assert (config.LOG_DIRECTORY / "my_custom_name.jsonl").exists()
    mock_auto_rename.assert_not_called()


def test_perform_interactive_chat_finalization_no_memory(
    session_state, fake_fs, mocker
):
    """Tests that memory consolidation is skipped when memory is disabled."""
    session_state.memory_enabled = False
    session_state.history = [utils.construct_user_message("openai", "test", [])]
    mocker.patch("aicli.session_manager.prompt", side_effect=EOFError)
    mock_consolidate = mocker.patch("aicli.commands.consolidate_session_into_memory")

    perform_interactive_chat(session_state, "test_session")
    mock_consolidate.assert_not_called()


def test_perform_interactive_chat_finalization_force_quit(
    session_state, fake_fs, mocker
):
    """Tests that finalization steps are skipped on a force quit."""
    session_state.force_quit = True
    session_state.history = [utils.construct_user_message("openai", "test", [])]
    mocker.patch("aicli.session_manager.prompt", side_effect=EOFError)
    mock_consolidate = mocker.patch("aicli.commands.consolidate_session_into_memory")
    mock_rename_log = mocker.patch("aicli.commands.rename_session_log")

    perform_interactive_chat(session_state, "test_session")
    mock_consolidate.assert_not_called()
    mock_rename_log.assert_not_called()


def test_perform_interactive_chat_finalization_debug_log(
    session_state, fake_fs, mocker
):
    """Tests that a debug log is saved when debug mode is active."""
    session_state.debug_active = True
    session_state.session_raw_logs = [{"request": "data1"}, {"response": "data2"}]
    mocker.patch("aicli.session_manager.prompt", side_effect=EOFError)

    perform_interactive_chat(session_state, "test_session")

    debug_log_path = config.LOG_DIRECTORY / "debug_test_session.jsonl"
    assert debug_log_path.exists()
    with open(debug_log_path) as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"request": "data1"}


def test_perform_interactive_chat_log_write_error(session_state, mocker, caplog):
    """Tests that a log write error is handled gracefully."""
    mocker.patch("aicli.session_manager.prompt", side_effect=["Hello", EOFError])
    mocker.patch("aicli.api_client.perform_chat_request", return_value=("Hi", {}))
    # Mock 'open' to raise an OSError specifically for the log file
    mocker.patch("builtins.open", side_effect=OSError("Disk full"))

    with caplog.at_level(logging.WARNING):
        perform_interactive_chat(session_state, "test_session")
        assert "Could not write to session log file: Disk full" in caplog.text

    # Verify that the session history was still updated
    assert len(session_state.history) == 2


def test_perform_interactive_chat_lifecycle(session_state, fake_fs, mocker):
    """
    Tests the main interactive chat loop from start to finish, including
    user input, API calls, history updates, logging, and finalization.
    """
    mock_prompt = mocker.patch(
        "aicli.session_manager.prompt",
        side_effect=["Hello, AI!", EOFError],
    )
    mock_api_call = mocker.patch(
        "aicli.api_client.perform_chat_request",
        return_value=(
            "Hello, User!",
            {"prompt": 10, "completion": 5, "total": 15},
        ),
    )
    mock_consolidate = mocker.patch("aicli.commands.consolidate_session_into_memory")
    mock_rename_log = mocker.patch("aicli.commands.rename_session_log")

    # Call with session_name=None to trigger auto-renaming logic
    perform_interactive_chat(session_state, None)

    assert mock_prompt.call_count == 2
    mock_api_call.assert_called_once()
    call_kwargs = mock_api_call.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["system_prompt"] == "Base prompt."
    messages_sent = call_kwargs["messages_or_contents"]
    assert len(messages_sent) == 1
    assert utils.extract_text_from_message(messages_sent[0]) == "Hello, AI!"
    assert len(session_state.history) == 2
    assert utils.extract_text_from_message(session_state.history[0]) == "Hello, AI!"
    assert utils.extract_text_from_message(session_state.history[1]) == "Hello, User!"

    # Since the name is generated with a timestamp, we just check that a log file exists
    log_files = list(config.LOG_DIRECTORY.glob("chat_*.jsonl"))
    assert len(log_files) == 1
    with open(log_files[0]) as f:
        log_data = json.load(f)
    assert log_data["model"] == "gpt-4o-mini"
    assert utils.extract_text_from_message(log_data["prompt"]) == "Hello, AI!"
    assert utils.extract_text_from_message(log_data["response"]) == "Hello, User!"
    assert log_data["tokens"]["total"] == 15
    mock_consolidate.assert_called_once()
    mock_rename_log.assert_called_once()
