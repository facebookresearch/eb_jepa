.ONESHELL:
.PHONY: help
.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

run_image_jepa: ## Run the image JEPA example
	uv run python examples/image_jepa/main.py

run_video_jepa: ## Run the video JEPA example
	uv run python examples/video_jepa/main.py

run_ac_video_jepa: ## Run the action-conditioned video JEPA example
	uv run python examples/ac_video_jepa/main.py
