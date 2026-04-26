IMAGE_NAME := danefedov-dev
CONTAINER_NAME := danefedov-dev-$(shell date +%Y-%m-%d-%H-%M)
TENSORBOARD_PORT ?= 6006

.PHONY: docker-run docker-build tensorboard

docker-run: docker-build
	docker run -d \
		--name "$(CONTAINER_NAME)" \
		-p $(TENSORBOARD_PORT):$(TENSORBOARD_PORT) \
		-v "$(CURDIR)":/app \
		-w /app \
		--gpus all \
		"$(IMAGE_NAME)" \
		tail -f /dev/null

docker-build:
	docker build -t $(IMAGE_NAME) .

tensorboard:
	uv run tensorboard --logdir "$(LOGDIR)" --host 0.0.0.0 --port $(TENSORBOARD_PORT)
