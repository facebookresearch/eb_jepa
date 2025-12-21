## paths
.ONESHELL:
.PHONY: help
.DEFAULT_GOAL := help

## print a help msg to display the comments
help:
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

USER := $(shell whoami)
PWD := $(shell pwd)
ROOT := $(shell cd ..;  pwd)

run_example_video_jepa:
	uv run python apps/example_video_jepa/main.py

run_example_video_jepa_chunked_vc:
	uv run python examples/video_jepa/main.py

run_example_ac_video_jepa:
	uv run python examples/ac_video_jepa/main.py

run_example_text_multistep:
	uv run torchrun --standalone --nproc_per_node=8 examples/text_multistep/main.py
