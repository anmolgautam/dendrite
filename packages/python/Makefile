.PHONY: test lint typecheck format ci clean

test:
	pytest tests/ -v --cov=src/dendrite --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src/dendrite/

format:
	ruff format .
	ruff check --fix .

ci: lint typecheck test

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
