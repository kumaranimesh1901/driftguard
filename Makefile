.PHONY: setup up down test lint seed demo queries
setup:
	python3 -m pip install -r requirements-dev.txt
up:
	docker compose up --build
down:
	docker compose down
test:
	python3 -m pytest -q
lint:
	python3 -m ruff check app tests
seed:
	python3 scripts/seed_demo.py
demo:
	python3 scripts/run_demo.py
queries:
	python3 scripts/run_queries.py
