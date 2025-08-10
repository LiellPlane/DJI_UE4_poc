build:
	docker build -t image-processor .

run:
	docker compose up

deploy-infra:
	terraform init
	terraform apply -auto-approve

test:
	uv run -m pytest --cov=app tests/ --cov-report=term-missing

clean:
	docker compose down
	docker rmi image-processor

lint:
	flake8 app/
	black --check app/