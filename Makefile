SRC_DIRS := salstm

# -rc recursively looks for python files in an input dir
run-isort:
	isort -rc $(SRC_DIRS)

run-pep8:
	python -m autopep8 --max-line-length 100 --recursive --in-place $(SRC_DIRS)

run-pylint:
	python3 -m pylint --rcfile=.pylintrc $(SRC_DIRS)

run-flake:
	python3 -m flake8 --ignore E501, W503 $(SRC_DIRS)


pre-commit: run-isort run-pep8 run-pylint #run-flake
