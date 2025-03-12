import pytest
from unittest.mock import MagicMock, patch, mock_open
from agent_log.agent_log import (
    validate_input,
    filter_content,
    parse_logs,
    analyze_phrases,
    analyze_errors,
    generate_report,
    format_report,
    analyze_log_file,
    LogAnalyzerState,
)

# Sample test data
SAMPLE_LOG_CONTENT = """
[2023-10-01 12:00:00] INFO - Agent Response: "Hello, how can I help you?"
[2023-10-01 12:01:00] ERROR - Failed to process request
[2023-10-01 12:02:00] WARNING - High memory usage detected
[2023-10-01 12:03:00] INFO - Agent Response: "Please provide more details."
"""

INVALID_LOG_CONTENT = "This is not a valid log format."

# Mock LLM responses
MOCK_PHRASE_ANALYSIS_RESPONSE = """
{
    "frequent_phrases": [
        {"phrase": "Hello, how can I help you?", "count": 1},
        {"phrase": "Please provide more details.", "count": 1}
    ]
}
"""

MOCK_ERROR_ANALYSIS_RESPONSE = """
{
    "common_errors": [
        {"error_type": "Failed to process request", "count": 1}
    ]
}
"""

MOCK_WORKFLOW_RESULT = {
    "final_report": {
        "log_summary": {"info_count": 2, "error_count": 1, "warning_count": 1},
        "frequent_phrases": [{"phrase": "Hello, how can I help you?", "count": 1}],
        "common_errors": [{"error_type": "Failed to process request", "count": 1}],
    }
}



# Test `validate_input`
def test_validate_input_valid():
    assert validate_input(SAMPLE_LOG_CONTENT) is True


def test_validate_input_invalid():
    assert validate_input(INVALID_LOG_CONTENT) is False


def test_validate_input_empty():
    assert validate_input("") is False


# Test `filter_content`
def test_filter_content():
    assert filter_content("This is a badword1 test.") == "This is a *** test."
    assert filter_content("No inappropriate words here.") == "No inappropriate words here."


# Test `parse_logs`
def test_parse_logs():
    state: LogAnalyzerState = {
        "log_content": SAMPLE_LOG_CONTENT,
        "info_messages": [],
        "error_messages": [],
        "warning_messages": [],
        "agent_responses": [],
        "log_summary": {},
        "frequent_phrases": [],
        "common_errors": [],
        "final_report": {},
    }

    result = parse_logs(state)

    assert len(result["info_messages"]) == 2
    assert len(result["error_messages"]) == 1
    assert len(result["warning_messages"]) == 1
    assert len(result["agent_responses"]) == 2
    assert result["log_summary"] == {"info_count": 2, "error_count": 1, "warning_count": 1}


def test_parse_logs_invalid_input():
    state: LogAnalyzerState = {
        "log_content": INVALID_LOG_CONTENT,
        "info_messages": [],
        "error_messages": [],
        "warning_messages": [],
        "agent_responses": [],
        "log_summary": {},
        "frequent_phrases": [],
        "common_errors": [],
        "final_report": {},
    }

    with pytest.raises(ValueError, match="Invalid log content"):
        parse_logs(state)


# Test `analyze_phrases`
@patch("agent_log.agent_log.llm")
def test_analyze_phrases(mock_llm):
    # Mock the LLM response
    mock_llm.return_value = MagicMock(content=MOCK_PHRASE_ANALYSIS_RESPONSE)

    state: LogAnalyzerState = {
        "log_content": SAMPLE_LOG_CONTENT,
        "info_messages": ["Agent Response: 'Hello, how can I help you?'"],
        "error_messages": [],
        "warning_messages": [],
        "agent_responses": ["Hello, how can I help you?"],
        "log_summary": {},
        "frequent_phrases": [],
        "common_errors": [],
        "final_report": {},
    }

    result = analyze_phrases(state)

    assert len(result["frequent_phrases"]) == 2
    assert result["frequent_phrases"][0]["phrase"] == "Hello, how can I help you?"
    assert result["frequent_phrases"][0]["count"] == 1


# Test `analyze_errors`
@patch("agent_log.agent_log.llm")
def test_analyze_errors(mock_llm):
    # Mock the LLM response
    mock_llm.return_value = MagicMock(content=MOCK_ERROR_ANALYSIS_RESPONSE)

    state: LogAnalyzerState = {
        "log_content": SAMPLE_LOG_CONTENT,
        "info_messages": [],
        "error_messages": ["Failed to process request"],
        "warning_messages": [],
        "agent_responses": [],
        "log_summary": {},
        "frequent_phrases": [],
        "common_errors": [],
        "final_report": {},
    }

    result = analyze_errors(state)

    assert len(result["common_errors"]) == 1
    assert result["common_errors"][0]["error_type"] == "Failed to process request"
    assert result["common_errors"][0]["count"] == 1


# Test `generate_report`
def test_generate_report():
    state: LogAnalyzerState = {
        "log_content": SAMPLE_LOG_CONTENT,
        "info_messages": [],
        "error_messages": [],
        "warning_messages": [],
        "agent_responses": [],
        "log_summary": {"info_count": 2, "error_count": 1, "warning_count": 1},
        "frequent_phrases": [{"phrase": "Hello, how can I help you?", "count": 1}],
        "common_errors": [{"error_type": "Failed to process request", "count": 1}],
        "final_report": {},
    }

    result = generate_report(state)

    assert result["final_report"]["log_summary"] == state["log_summary"]
    assert result["final_report"]["frequent_phrases"] == state["frequent_phrases"]
    assert result["final_report"]["common_errors"] == state["common_errors"]


# Test `format_report`
def test_format_report():
    report = {
        "log_summary": {"info_count": 2, "error_count": 1, "warning_count": 1},
        "frequent_phrases": [{"phrase": "Hello, how can I help you?", "count": 1}],
        "common_errors": [{"error_type": "Failed to process request", "count": 1}],
    }

    formatted = format_report(report)

    assert "INFO messages: 2" in formatted
    assert "ERROR messages: 1" in formatted
    assert "WARNING messages: 1" in formatted
    assert "Hello, how can I help you?" in formatted
    assert "Failed to process request" in formatted


# Test `analyze_log_file`
@patch("agent_log.agent_log.create_workflow")
def test_analyze_log_file_success(mock_create_workflow):
    mock_workflow = MagicMock()
    mock_workflow.invoke.return_value = MOCK_WORKFLOW_RESULT
    mock_create_workflow.return_value = mock_workflow

    with patch("builtins.open", mock_open(read_data=SAMPLE_LOG_CONTENT)):
        report, formatted_report = analyze_log_file("dummy_log_file.txt")

        assert report["log_summary"]["info_count"] == 2
        assert "INFO messages: 2" in formatted_report
        assert "Hello, how can I help you?" in formatted_report
        assert "Failed to process request" in formatted_report

def test_analyze_log_file_file_not_found():
    with pytest.raises(FileNotFoundError) as e:
        analyze_log_file("nonexistent_file.txt")
