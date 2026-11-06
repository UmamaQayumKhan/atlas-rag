.PHONY: help install dev test lint eval eval-answer sweep gate ingest run ui docker clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## install runtime dependencies
	pip install -r requirements.txt

dev: ## install dev dependencies
	pip install -r requirements-dev.txt

test: ## run the test suite (offline, no keys needed)
	PYTHONPATH=src pytest -q

lint: ## ruff + mypy
	ruff check src eval tests
	mypy src/atlas || true

ingest: ## index data/corpus
	PYTHONPATH=src python -m atlas.cli ingest

run: ## start the API on :8000
	PYTHONPATH=src uvicorn atlas.api:app --reload --port 8000

ui: ## start the Streamlit UI on :8501
	streamlit run ui/app.py

eval: ## retrieval eval (fast, offline)
	LLM_PROVIDER=offline EMBEDDING_PROVIDER=offline VECTOR_BACKEND=memory \
	  python eval/run_eval.py --mode retrieval

eval-answer: ## full answer eval (needs a provider key)
	python eval/run_eval.py --mode answer --judge

sweep: ## compare chunking strategies
	LLM_PROVIDER=offline EMBEDDING_PROVIDER=offline python eval/run_eval.py --sweep

gate: ## the CI quality gate
	LLM_PROVIDER=offline EMBEDDING_PROVIDER=offline VECTOR_BACKEND=memory \
	  python eval/run_eval.py --mode retrieval --gate-recall5 0.85 --gate-mrr 0.70

docker: ## bring up qdrant + api + ui
	docker compose up --build

clean:
	rm -rf .atlas_index .atlas_eval* .pytest_cache eval/report.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
