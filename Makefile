.PHONY : test
test:
	python -m pytest --disable-warnings .

.PHONY : lint
lint:
	ruff check --fix .
	ruff format .
	pyright .

.PHONY : publish
publish:
	poetry publish --build
