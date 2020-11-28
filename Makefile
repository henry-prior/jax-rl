init:
	poetry install

update: 
	poetry update

format:
	poetry run black jax_rl/ tests/

test:
	poetry run pytest tests/

