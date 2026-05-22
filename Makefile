.PHONY: install lint test train-local build-image

install:
	pip install -e ".[train,serve,dev]"

lint:
	ruff check . --fix
	ruff format .

test:
	pytest tests/ -v

train-local:
	python training/scripts/train.py \
		--config training/configs/local_smoke.yaml \
		--max_samples 50

train-full:
	python training/scripts/train.py \
		--config training/configs/full_train.yaml

build-image:
	docker build -f infra/docker/Dockerfile.serve -t vlm-action-captioning:latest .

serve-local:
	uvicorn serving.app.main:app --reload --port 8000