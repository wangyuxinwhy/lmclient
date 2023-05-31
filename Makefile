.PHONY : test
test:
	python -m pytest --disable-warnings .

.PHONY : lint
lint:
	isort .
	blue -l 128 .
	pyright .

.PHONY : publish
publish:
	poetry publish --build
