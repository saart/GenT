# Gen-T

Gen-T is a cutting-edge system designed to address the challenges of distributed tracing (DT) in microservice architectures by leveraging deep generative compression techniques. 
This repository includes the source code, test cases, and supplementary data for our paper.

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Introduction

Microservice architectures are widely adopted for building scalable and agile software systems. As these environments grow in complexity, the need for observability (o11y) tools to manage, diagnose, and optimize these systems becomes critical. Distributed tracing is a key component of o11y, providing an end-to-end view of requests as they travel through various services.

However, the cost of transmitting traces often leads to operators disabling or reducing tracing, particularly during the triage phase. This project addresses this issue by proposing a novel approach using deep generative compression to offer better fidelity-flexibility-cost tradeoffs for DT.

## Repository Structure

The repository is organized as follows:

```
GenT/
├── README.md
├── results/ # Contains results from Gen-T experiments
├── src/
│ ├── collector/ # Code related to data collection and normalization
│ ├── drivers/ # Drivers for different trace generation models
│ ├── fidelity/ # Fidelity analysis and data processing tools
│ ├── ml/ # Machine learning models and utilities
│ ├── gent_utils/ # Utility functions and constants for GenT
│ ├── paper/ # Supplementary materials related to the paper itself (e.g. figures, baselines)
│ ├── requirements.txt # Python dependencies
├── tests/
│ ├── collector/ # Test cases for collector components
│ ├── fidelity/ # Test cases for fidelity components
│ ├── ml/ # Test cases for ML components
│ ├── test_data/ # Test data used in test cases
├── traces/ # Directory containing trace data - not included in the GitHub repository
├── venv/ # Virtual environment directory
```


## Getting Started

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/installation/)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/saart/GenT.git
    cd GenT
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r src/requirements.txt
    ```

## Usage

### Downloading Traces from Jaeger

If you need to download traces from the Jaeger API, you can use the provided utility:

```python
from paper.benchmark.download_from_jaeger import download_traces_from_jaeger_for_all_services

download_traces_from_jaeger_for_all_services(target_dir=f"abs/path/to/otel/traces", jaeger_url="http://localhost:16686")
```

### Translating Jaeger Traces to GenT Format
Before running the GenTDriver on a directory containing raw OTEL spans, you need to parse them into a format that GenT can process using the following utility:
```python
from paper.benchmark.download_from_jaeger import translate_jaeger_to_gent

translate_jaeger_to_gent(from_dir=f"abs/path/to/otel/traces", to_dir=f"abs/target/path/to/gent/traces")
```

### Running the Training and Generation Process
To run the training and generation process, you can use the following script:

```python
from drivers.gent.gent_driver import GenTDriver

# Configure the GenTDriver with the desired configuration
config = GenTConfig(chain_length=2, iterations=3, tx_end=10000 + i, traces_dir="path/to/traces_dir")

# Initialize the driver
driver = GenTDriver(config)

# Train the model and generate traces
driver.train_and_generate()
```

#### Parameters
* chain_length: Length of the chain as described in the paper.
* iterations: Number of iterations for training.
* tx_end: number of traces to use for training.
* traces_dir: Directory where trace data is stored.

Ensure that you update the traces_dir with the appropriate path to your trace data directory.


## Contributing

We welcome contributions to Gen-T! Please fork the repository and submit pull requests for review.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

