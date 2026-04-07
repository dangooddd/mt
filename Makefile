IMAGE_NAME := danefedov-dev
CONTAINER_NAME := danefedov-dev-$(shell date +%Y%m%d-%H%M%S)

.PHONY: docker-run docker-build

docker-run: docker-build
	docker run -d \
		--name "$(CONTAINER_NAME)" \
		-v "$(CURDIR)":/app \
		-w /app \
		"$(IMAGE_NAME)" \
		tail -f /dev/null

docker-build:
	docker build -t $(IMAGE_NAME) .
