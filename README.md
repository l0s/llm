# LLM

[![Continuous Integration](https://github.com/l0s/llm/actions/workflows/ci.yml/badge.svg)](https://github.com/l0s/llm/actions/workflows/ci.yml)

This project contains the exercises from [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka.

## Formatting

```
uv run ruff format
```

## Tests

```
uv run pytest
```

Mutation Testing

```
uv run mut.py --target llm --unit-test tests --show-mutants --runner pytest --colored-output
```
These are not included in the continuous integration build as they take more than 10 minutes.