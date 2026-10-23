.PHONY: setup serve test build
setup:
	bash scripts/setup.sh
serve:
	bash scripts/serve.sh
test:
	PYTHONPATH=backend .venv/bin/python -m pytest
build:
	npm run build
