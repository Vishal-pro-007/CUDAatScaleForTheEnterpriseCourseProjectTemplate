.PHONY: install run clean

install:
	pip install numba opencv-python matplotlib numpy

run:
	python bin/main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
