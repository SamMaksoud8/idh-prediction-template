# Set the default goal to "help", so running `make` on its own shows the help menu.
.DEFAULT_GOAL := help

# Phony targets are commands, not files.
.PHONY: install lint format test clean help

# ====================================================================================
#  Development Workflow Targets
# ====================================================================================

auth:
	@echo "🔑 Using $(GOOGLE_APPLICATION_CREDENTIALS)"
	@test -f "$(GOOGLE_APPLICATION_CREDENTIALS)" || (echo "❌ Missing key file"; exit 1)
	@gcloud auth activate-service-account --key-file="$(GOOGLE_APPLICATION_CREDENTIALS)"

build: ## Build the Docker image for local development
	docker build -t idh-csv-predictor:local .

run: ## Run the Docker container locally
	docker run --rm -p 8080:8080 \
	-e GOOGLE_APPLICATION_CREDENTIALS="$(abspath $(GOOGLE_APPLICATION_CREDENTIALS))" \
	-v "$(abspath $(GOOGLE_APPLICATION_CREDENTIALS)):$(abspath $(GOOGLE_APPLICATION_CREDENTIALS)):ro" \
	idh-csv-predictor:local

run-app: ## Run the Docker container locally
	docker run --rm -p 8080:8080 \
	-e GOOGLE_APPLICATION_CREDENTIALS="$(abspath $(GOOGLE_APPLICATION_CREDENTIALS))" \
	-v "$(abspath $(GOOGLE_APPLICATION_CREDENTIALS)):$(abspath $(GOOGLE_APPLICATION_CREDENTIALS)):ro" \
	idh-csv-predictor:local \
	python src/idh/app/csv_prediction.py --share

install: ## Install project dependencies from requirements.txt
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

lint: ## Check for code style and errors with Ruff
	@echo "🔍 Running linter..."
	ruff check .

format: ## Automatically format code with Ruff
	@echo "🎨 Formatting code..."
	ruff format .

test: ## Run tests with pytest
	@echo "🧪 Running tests..."
	pytest

clean: ## Clean up temporary Python files
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .ruff_cache

# ====================================================================================
#  Help Target - This is a self-documenting trick!
# ====================================================================================

help: ## ✨ Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
