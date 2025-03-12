.PHONY: tests
project = agent_log

install-poetry:
	pip install --no-cache-dir -U pip
	pip install "poetry>=2.0"

install-dev:
	poetry lock
	poetry install --all-extras

tests:
	poetry run pytest tests/test_agent_log.py -v
