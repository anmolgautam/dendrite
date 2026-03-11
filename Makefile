.PHONY: ci ci-python test lint typecheck format clean

# Default: run CI for all packages
ci: ci-python

# Per-package CI
ci-python:
	$(MAKE) -C packages/python ci

# Shortcuts that delegate to Python package
test:
	$(MAKE) -C packages/python test

lint:
	$(MAKE) -C packages/python lint

typecheck:
	$(MAKE) -C packages/python typecheck

format:
	$(MAKE) -C packages/python format

clean:
	$(MAKE) -C packages/python clean
