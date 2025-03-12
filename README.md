# AI Agent Log Analyzer

## Overview
The **AI Agent Log Analyzer** is a Python tool designed to analyze AI agent logs, extract key insights, and generate structured reports. It parses log files to count INFO, ERROR, and WARNING messages, identifies frequently used agent responses, and categorizes common errors. The system utilizes OpenAI's **GPT-4o** via LangChain to enhance phrase and error analysis.

## Features
- Parses logs to extract **INFO, ERROR, and WARNING** messages.
- Identifies and counts **frequently used AI responses**.
- Analyzes error messages to categorize common errors.
- Uses **GPT-4o** for structured phrase and error analysis.
- Generates reports in **formatted text** and **JSON**.

## Project Structure
```
agent_log/
│── agent_log.py        # Main log analysis script
│── log.txt             # Sample log file
│── tests/              # Test suite
│   ├── test_agent_log.py  # Unit tests for log analyzer
│── .venv/              # Virtual environment (if used)
│── .env                # Environment variables (OpenAI API key)
│── .gitignore          # Git ignore file
│── Makefile            # Make commands for installation & testing
│── poetry.lock         # Poetry dependency lock file
│── pyproject.toml      # Poetry dependency manager file
│── README.md           # Documentation
│── report.json         # Example structured report
│── report.txt          # Example formatted report
```

## Installation
### **1. Clone the Repository**
```sh
 git clone <repository_url>
 cd agent_log
```

### **2. Set Up Virtual Environment**
```sh
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### **3. Install Dependencies**
#### **Using Poetry**
```sh
make install-poetry
make install-dev
```

### **4. Set OpenAI API Key**
Create a `.env` file in the project root and add your API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage
### **Run the Log Analyzer**
```sh
poetry run python agent_log/agent_log.py agent_log/log.txt
```
#### **Save Results to a File**
```sh
poetry run python agent_log/agent_log.py agent_log/log.txt --output report.txt --json report.json
```

## Running Tests
```sh
make tests
```
Or run manually with:
```sh
poetry run pytest tests/test_agent_log.py -v
```

## Example Output
```
Log Summary:
- INFO messages: 42
- ERROR messages: 8
- WARNING messages: 5

Top AI Responses:
1. "Hello! How can I help you today?" (12 times)
2. "I'm sorry, I didn't understand that." (7 times)
3. "Please provide more details." (5 times)

Most Common Errors:
1. Model Timeout after 5000ms (3 times)
2. API Connection Failure (2 times)
```

## License
This project is licensed under the MIT License.

