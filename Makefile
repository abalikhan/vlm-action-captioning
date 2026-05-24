.PHONY: install lint test train-local build-image run-container run-container-cpu push-ecr

IMAGE_NAME=vlm-action-captioning
IMAGE_TAG=latest

install:
	uv sync --extra train --extra serve --extra dev

lint:
	ruff check . --fix
	ruff format .

test:
	.venv/bin/pytest tests/ -v

train-local:
	python training/scripts/train.py \
		--config training/configs/local_smoke.yaml \
		--max_samples 50

train-full:
	python training/scripts/train.py \
		--config training/configs/full_train.yaml

serve-local:
	uvicorn serving.app.main:app --reload --port 8000

build-image:
	mkdir -p model_weights
	docker build \
		-f infra/docker/Dockerfile.serve \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		.

run-container:
	docker run --gpus all \
		-p 8000:8000 \
		-e HF_HOME=/app/.cache/huggingface \
		$(IMAGE_NAME):$(IMAGE_TAG)

run-container-cpu:
	docker run \
		-p 8000:8000 \
		$(IMAGE_NAME):$(IMAGE_TAG)

push-ecr:
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_REGISTRY)
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(ECR_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push $(ECR_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
modal-download-model:
	modal run serving/modal/app.py::download_model

modal-deploy:
	modal deploy serving/modal/app.py

modal-serve-dev:
	modal serve serving/modal/app.py
deploy-spaces:
	cd hf-space-tmp && git push || \
	(git clone https://huggingface.co/spaces/your-hf-username/vlm-action-captioning hf-space-tmp && \
	cp serving/gradio/app.py hf-space-tmp/ && \
	cp serving/gradio/requirements.txt hf-space-tmp/ && \
	cd hf-space-tmp && git add . && git commit -m "update demo" && git push)