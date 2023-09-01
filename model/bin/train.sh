#!/bin/bash

# Usage: ./bin/train.sh --model-name bigscience/bloom-3b --repo mickume/harry_potter_tiny

python -m trainer.train $@