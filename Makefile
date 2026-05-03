.PHONY: install test lint format health run-stm

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-slow:
	pytest tests/ -v -m slow

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

health:
	python -m qrc_thresher.cli health

run-stm:
	python -m qrc_thresher.cli run stm --config configs/alpha_lite.yaml
