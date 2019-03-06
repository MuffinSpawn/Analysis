#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# python -m unittest ${SCRIPT_DIR}/../test/family_tree/test_api.py
python -m unittest discover -v -s ${SCRIPT_DIR}/..
