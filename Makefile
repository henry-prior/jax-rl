init:
	poetry install

update:
	poetry update

format:
	poetry run black jax_rl/ tests/

test:
	poetry run pre-commit run --all-files
	poetry run pytest tests/
