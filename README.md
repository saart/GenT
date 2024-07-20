# Gen-T

Gen-T is a system designed to reduce the triage cost of distributed tracing (DT) in microservice architectures by leveraging deep generative compression techniques.
This repository includes the source code, test cases, and supplementary data for our paper.

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Use in Production](#use-in-production)
- [Replicate Experiments from the Paper](#replicate-experiments-from-the-paper)
  - [Demo Application Experiments](#demo-application-experiments)
  - [PandoraTrace Benchmark](#pandoratrace-benchmark)
  - [Scripts for Paper Reproduction](#scripts-for-paper-reproduction)
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
│ ├── pandora_trace/ # Our framework to create tracing data from various applications and different error types
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

## Use in production

To use Gen-T in production, you need to incorporate two main components:

1. **Compression Mechanism at the Collector**: This component batches the incoming spans. We recommend using a batch size of around 10,000 traces. Given a batch of OTEL traces, create a compressed model using the following steps:
```python
from drivers.gent.gent_driver import GenTDriver
from pandora_trace.jaeger_to_gent import translate_jaeger_to_gent

# preprocess the OTEL traces into GenT format
translate_jaeger_to_gent(from_dir=f"abs/path/to/otel/traces", to_dir=f"abs/target/path/to/gent/traces")

# Initialize the driver and train the model
driver = GenTDriver(GenTConfig(traces_dir="abs/target/path/to/gent/traces"))
driver.train()  # note: it takes a while to train the model. Consider around 1 minute per 10,000 traces and 10 iterations in a machine with a single GPU.

to_send = driver.get_model_gzip_file()  # returns the path to GenT model
```
2. **Decompression Mechanism at the Backend**: This component generates synthetic traces from the model. Given a compressed model, create a GenTDriver and generate traces using the following steps:
```python
from drivers.gent.gent_driver import GenTDriver

driver = GenTDriver(GenTConfig(traces_dir="abs/target/path/to/gent/traces"))
driver.load_model_gzip_file("path/to/GenT/model.zip")
driver.generate()  # note: it takes a while to train the model. Consider around 1 minute, as mentioned above.
# the generated traces will be stored in the traces_dir
```
As discussed in the paper, GenTConfig is highly configurable and offers different trade-offs to better fit the target application. Ensure you use the same parameters in both the compression and decompression processes.

Note: This repository represents ongoing academic research. Please take the necessary precautions when using it in production environments. Use at your own risk.

## Replicate Experiments from the Paper

The paper runs experiments in two settings. The first uses existing Jaeger traces from demo applications (we used HotROD and WildRydes). The second generates new traces with PandoraTrace, a new observability benchmark suite that we developed to provide the research community greater granularity and control in evaluating incidents in microservice environments.

### Demo Application Experiments

#### Downloading Traces from Jaeger

If you need to download traces from the Jaeger API, you can use the provided utility:

```python
from pandora_trace.jaeger_to_gent import download_traces_from_jaeger_for_all_services

download_traces_from_jaeger_for_all_services(target_dir=f"abs/path/to/otel/traces", jaeger_url="http://localhost:16686")
```

#### Translating Jaeger Traces to GenT Format
Before running the GenTDriver on a directory containing raw OTEL spans, you need to parse them into a format that GenT can process using the following utility:
```python
from pandora_trace.jaeger_to_gent import translate_jaeger_to_gent

translate_jaeger_to_gent(from_dir=f"abs/path/to/otel/traces", to_dir=f"abs/target/path/to/gent/traces")
```

#### Running the Training and Generation Process
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

##### Parameters
* chain_length: Length of the chain as described in the paper.
* iterations: Number of iterations for training.
* tx_end: number of traces to use for training.
* traces_dir: Directory where trace data is stored.

Ensure that you update the traces_dir with the appropriate path to your trace data directory.



## PandoraTrace Benchmark

PandoraTrace is a benchmark designed to create tracing data from various applications and different error types.

### Structure

Users of PandoraTrace can programmatically produce (and probe) traces that exhibit incidents by specifying four key components: Applications, Request Patterns, Incident Types, and Queries.

* **Applications**: We used the microservices applications from the DeathStarBench suite: socialNetwork, hotelReservation, and mediaMicroservices.
* **Request Patterns**: To simulate realistic and diverse traffic patterns, we utilized RESTler, a stateful REST API fuzzer. RESTler generates a broad spectrum of trace request patterns with varying properties, effectively mimicking the unpredictable nature of real user interactions with these services.
* **Incident Types**: We generated traces simulating 10 specific incidents, ranging from induced delays to internal errors in third-party services. These incidents are induced in the running application using various tools, such as the “stress” apt package to simulate CPU and memory bottlenecks, and “iproute2” for introducing network delays.
* **Queries**: We use a set of 10 queries derived from TraceQL, Jaeger, and PromQL.

### Using the Benchmark

Using this benchmark involves two main steps:
1. Creating the raw traces. Each created directory holds traces for a single DeathStarBench application and an error.
2. Comparing raw traces to synthetic traces. We assume that these traces are provided as two SQL tables. Our output is a single number: the average Wasserstein distance across all the generated queries.

#### Creating Raw Traces

* `run_benchmark.create_baseline`: Creates a set of traces of a specific application without any incidents.
* `run_benchmark.main`:
  * Runs each DeathStarBench application.
  * Adds an incident.
  * Creates traffic using RESTler (based on the Swagger docs).
  * Downloads the Jaeger traces and translates them into GenT format.
  * Merges the erroneous traces with the baseline (benign) data using a distribution derived from an exponential random variable.

#### Comparing Results

The API function `run_template` takes a dictionary that maps attributes to values and a list of SQL queries parameterized by these attributes, additionally it takes the name of two SQL tables and a DB cursor. It formats the queries using the given attributes, runs the resulting queries on the input tables, and compares the queries results using the Wasserstein distance. It returns the average distance.


## Scripts for Paper Reproduction

1. All the traces that were used in the paper can be downloaded from https://gen-t-code.s3.us-west-2.amazonaws.com/traces.zip.
2. The file `src/paper/ml_ops.py` executes the different ablation configurations used in the paper: `chain_length`, `ctgan_dim`, etc. 
3. The file `src/paper/adaption_experiment.py` executes the simulation where the traces' properties have changed every batch.
4. The file `src/fidelity/raw_sql.py` is used to calculate fidelity metrics using SQLite DB that holds synthetic and raw traces. The function `fill_data` fills the database with the raw/synthetic traces from the given directory. Functions like `trigger_correlation`, `monitor_errors`, `bottlenecks_by_time_range`, calculate a specific fidelity metric given syn/raw tables. Functions like `ctgan_gen_dim`, `rolling_experiment` are used to generate the final fidelity data for the figures in the paper.
5. The file `src/paper/figures.py` generates the figures used in the paper, based on the output from the previous step. In the `main` function, comment in/out the desired figures to generate.


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

## References

If you use this code in a publication, please cite the following work: 

@inproceedings{tochner2023gen,\
    title={Gen-T: Reduce Distributed Tracing Operational Costs Using Generative Models},\  
    author={Tochner, Saar and Fanti, Giulia and Sekar, Vyas},\
    booktitle={Temporal Graph Learning Workshop@ NeurIPS 2023},\
    year={2023}\
}
