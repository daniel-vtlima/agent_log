import re
import json
import argparse
import os
from typing import Dict, List, TypedDict
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv  # Import dotenv

from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# Define Pydantic models for structured outputs
class LogSummary(BaseModel):
    info_count: int = Field(description="Number of INFO messages")
    error_count: int = Field(description="Number of ERROR messages")
    warning_count: int = Field(description="Number of WARNING messages")

class PhraseFrequency(BaseModel):
    phrase: str = Field(description="The exact phrase or response pattern")
    count: int = Field(description="Approximate frequency of this phrase in responses")

class PhraseAnalysisResult(BaseModel):
    frequent_phrases: List[PhraseFrequency] = Field(
        description="List of frequently occurring phrases in agent responses"
    )

class ErrorFrequency(BaseModel):
    error_type: str = Field(description="Type or category of error")
    count: int = Field(description="Approximate frequency of this error")

class ErrorAnalysisResult(BaseModel):
    common_errors: List[ErrorFrequency] = Field(
        description="List of common error types with their frequencies"
    )

# Define graph state as TypedDict
class LogAnalyzerState(TypedDict):
    log_content: str
    info_messages: List[str]
    error_messages: List[str]
    warning_messages: List[str]
    agent_responses: List[str]
    log_summary: Dict
    frequent_phrases: List[Dict]
    common_errors: List[Dict]
    final_report: Dict

# Initialize LLM - using GPT-4o
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")  # Use the API key from .env
)

def validate_input(log_content: str) -> bool:
    """
    Validate the input log content to ensure it is properly formatted and safe.
    """
    # Basic check for non-empty content
    if not log_content.strip():
        return False

    # Check for common log patterns
    log_pattern = r'\[(.*?)\] (INFO|ERROR|WARNING) - (.*)'
    if not re.search(log_pattern, log_content):
        return False

    return True

def filter_content(text: str) -> str:
    """
    Filter out inappropriate content from the text.
    """
    inappropriate_words = ["badword1", "badword2"]  # Add your list of inappropriate words
    for word in inappropriate_words:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '***', text, flags=re.IGNORECASE)

    return text

def parse_logs(state: LogAnalyzerState) -> LogAnalyzerState:
    """
    Parse log file and extract relevant information.
    """
    log_content = state["log_content"]

    if not validate_input(log_content):
        raise ValueError("Invalid log content")

    log_pattern = r'\[(.*?)\] (INFO|ERROR|WARNING) - (.*)'

    info_messages = []
    error_messages = []
    warning_messages = []
    agent_responses = []

    for line in log_content.strip().split("\n"):
        match = re.match(log_pattern, line)
        if match:
            timestamp, log_level, message = match.groups()

            # Filter the message content
            message = filter_content(message)

            if log_level == "INFO":
                info_messages.append(message)
                # Check if it's an agent response
                response_match = re.search(r'Agent Response: "(.*?)"', message)
                if response_match:
                    agent_responses.append(filter_content(response_match.group(1)))
            elif log_level == "ERROR":
                error_messages.append(message)
            elif log_level == "WARNING":
                warning_messages.append(message)

    # Create structured summary
    log_summary = {
        "info_count": len(info_messages),
        "error_count": len(error_messages),
        "warning_count": len(warning_messages)
    }

    return {
        **state,
        "info_messages": info_messages,
        "error_messages": error_messages,
        "warning_messages": warning_messages,
        "agent_responses": agent_responses,
        "log_summary": log_summary
    }

def analyze_phrases(state: LogAnalyzerState) -> LogAnalyzerState:
    """
    Analyze agent responses to identify common phrases and patterns.
    Uses structured output with Pydantic.
    """
    agent_responses = state["agent_responses"]

    if not agent_responses or len(agent_responses) == 0:
        return {**state, "frequent_phrases": []}

    # Create the parser
    parser = PydanticOutputParser(pydantic_object=PhraseAnalysisResult)

    # Create the prompt template
    template = """
    Analyze these AI agent responses to identify the most frequently used phrases or patterns:

    RESPONSES:
    {responses}

    Your task is to:
    1. Identify common phrases, response patterns, or semantic templates
    2. Estimate how many times each appears in the responses
    3. Focus on meaningful patterns, not just common words
    4. DO NOT MAKE UP INFORMATION, JUST USE THE CONTEXT

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Format the prompt with our responses
    formatted_prompt = prompt.format(
        responses="\n".join(agent_responses)
    )

    # Invoke the LLM with structured output parsing
    result = llm(formatted_prompt)  # Call the LLM directly

    try:
        parsed_result = parser.parse(result.content)
        return {**state, "frequent_phrases": [p.model_dump() for p in parsed_result.frequent_phrases]}
    except ValidationError as e:
        print(f"Validation error parsing phrase analysis: {e}")
        print(f"Raw response: {result.content}")
        return {**state, "frequent_phrases": []}
    except Exception as e:
        print(f"Error parsing phrase analysis: {e}")
        print(f"Raw response: {result.content}")
        return {**state, "frequent_phrases": []}

def analyze_errors(state: LogAnalyzerState) -> LogAnalyzerState:
    """
    Analyze error messages to identify common error types and their frequencies.
    Uses structured output with Pydantic.
    """
    error_messages = state["error_messages"]

    if not error_messages or len(error_messages) == 0:
        return {**state, "common_errors": []}

    # Create the parser
    parser = PydanticOutputParser(pydantic_object=ErrorAnalysisResult)

    # Create the prompt template
    template = """
    Analyze these error messages to identify common error types and their frequencies:

    ERROR MESSAGES:
    {errors}

    Your task is to:
    1. Group similar errors into categories or types
    2. Estimate how many times each error type appears
    3. Focus on meaningful patterns that would help a developer fix issues
    4. DO NOT MAKE UP INFORMATION, JUST USE THE CONTEXT

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Format the prompt with our error messages
    formatted_prompt = prompt.format(
        errors="\n".join(error_messages)
    )

    # Invoke the LLM with structured output parsing
    result = llm(formatted_prompt)  # Call the LLM directly

    # Parse the result
    try:
        parsed_result = parser.parse(result.content)
        return {**state, "common_errors": [e.model_dump() for e in parsed_result.common_errors]}
    except ValidationError as e:
        print(f"Validation error parsing error analysis: {e}")
        print(f"Raw response: {result.content}")
        return {**state, "common_errors": []}
    except Exception as e:
        print(f"Error parsing error analysis: {e}")
        print(f"Raw response: {result.content}")
        return {**state, "common_errors": []}

def generate_report(state: LogAnalyzerState) -> LogAnalyzerState:
    """
    Generate the final report combining all analysis results.
    """
    report = {
        "log_summary": state["log_summary"],
        "frequent_phrases": state["frequent_phrases"],
        "common_errors": state["common_errors"]
    }

    return {**state, "final_report": report}

def create_workflow():
    """Create the LangGraph workflow."""
    workflow = StateGraph(LogAnalyzerState)

    workflow.add_node("parse_logs", parse_logs)
    workflow.add_node("analyze_phrases", analyze_phrases)
    workflow.add_node("analyze_errors", analyze_errors)
    workflow.add_node("generate_report", generate_report)

    workflow.add_edge("parse_logs", "analyze_phrases")
    workflow.add_edge("analyze_phrases", "analyze_errors")
    workflow.add_edge("analyze_errors", "generate_report")
    workflow.add_edge("generate_report", END)

    workflow.set_entry_point("parse_logs")

    return workflow.compile()

def format_report(report: Dict) -> str:
    """Format the structured report for human readability."""
    output = "Log Summary:\n"
    output += f"- INFO messages: {report['log_summary']['info_count']}\n"
    output += f"- ERROR messages: {report['log_summary']['error_count']}\n"
    output += f"- WARNING messages: {report['log_summary']['warning_count']}\n\n"

    output += "Common Agent Response Phrases:\n"
    if report["frequent_phrases"]:
        for i, item in enumerate(report["frequent_phrases"], 1):
            output += f"{i}. \"{item['phrase']}\" ({item['count']} times)\n"
    else:
        output += "No common phrases identified.\n"
    output += "\n"

    output += "Common Errors:\n"
    if report["common_errors"]:
        for i, item in enumerate(report["common_errors"], 1):
            output += f"{i}. {item['error_type']} ({item['count']} occurrences)\n"
    else:
        output += "No errors identified.\n"

    return output

def analyze_log_file(file_path: str):
    """Analyze a log file and return structured and formatted reports."""
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            log_content = f.read()

        # Initialize state
        initial_state: LogAnalyzerState = {
            "log_content": log_content,
            "info_messages": [],
            "error_messages": [],
            "warning_messages": [],
            "agent_responses": [],
            "log_summary": {},
            "frequent_phrases": [],
            "common_errors": [],
            "final_report": {}
        }

        # Run the workflow
        workflow = create_workflow()
        result = workflow.invoke(initial_state)

        # Get the structured report
        report = result["final_report"]

        # Generate formatted report
        formatted_report = format_report(report)

        return report, formatted_report
    except FileNotFoundError as e:
        error_msg = f"Error Log file does not exist at {file_path}: {str(e)}"
        print(error_msg)
        raise
    except Exception as e:
        error_msg = f"Error analyzing log file: {str(e)}"
        print(error_msg)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Agent Log Analyzer")
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument("--output", "-o", help="Output file for formatted report")
    parser.add_argument("--json", "-j", help="JSON output file for structured data")

    args = parser.parse_args()

    report, formatted_report = analyze_log_file(args.log_file)

    if args.json:
        with open(args.json, 'w', encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Structured report saved to {args.json}")

    if args.output:
        with open(args.output, 'w', encoding="utf-8") as f:
            f.write(formatted_report)
        print(f"Formatted report saved to {args.output}")
    else:
        print("\n" + formatted_report)
