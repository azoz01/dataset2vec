#!/bin/bash
set -e

echo "Running black"
black --line-length=79 dataset2vec/

echo "Running mypy"
mypy --install-types --ignore-missing-imports dataset2vec/

echo "Running flake8"
flake8 dataset2vec/
