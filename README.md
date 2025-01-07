# ml-model-benchmark-suite

A structured framework for training, evaluating, and comparing machine learning models across datasets with cross-validation, feature analysis, and experiment tracking.

## Installation

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

Or install the package in editable mode:

```bash
pip install -e .
```

## Usage

Run a benchmark experiment from a config file:

```bash
python main.py --config config/example.yaml
```

List available models:

```bash
python main.py --list-models
```

View experiment history:

```bash
python main.py --history
```

## Configuration

Experiments are driven by YAML or JSON config files. See `config/example.yaml` for a complete example.

## Author

DigitalNomad

Back to roasting beans ☕
