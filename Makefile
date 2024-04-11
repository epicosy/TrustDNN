.PHONY: clean virtualenv install install-dev test docker dist dist-upload

clean:
	find . -name '*.py[co]' -delete

install:
	pip install .

install-dev:
	pip install -e .[dev]

virtualenv:
	virtualenv --prompt '|> trustdnn <| ' env
	$(MAKE) install-dev
	@echo
	@echo "VirtualENV Setup Complete. Now run: source env/bin/activate"
	@echo

test:
	python -m pytest \
		-v \
		--cov=trustdnn \
		--cov-report=term \
		--cov-report=html:coverage-report \
		tests/

docker: clean
	docker build -t trustdnn:latest .

dist: clean
	rm -rf dist/*
	python setup.py sdist
	python setup.py bdist_wheel

dist-upload:
	twine upload dist/*
