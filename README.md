# INFO-SEDD: Continuous Time Markov Chains as Scalable Information Metrics Estimators
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Installation

To install the project, follow these steps:

1. **Create a Conda environment with Python 3.10**:

    ```bash
    conda create -n infosedd python=3.10
    ```

2. **Activate the Conda environment**:

    ```bash
    conda activate infosedd
    ```

3. **Install the project dependencies**:

    ```bash
    pip install -e .
    ```

## Usage

After installing the project, you can run the scripts provided in the repository.

To run an experiment with synthetic data do:
```bash
cd evaluation
bash run_experiment.sh
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{foresti2025info,
  title={INFO-SEDD: Continuous Time Markov Chains as Scalable Information Metrics Estimators},
  author={Foresti, Alberto and Franzese, Giulio and Michiardi, Pietro},
  journal={arXiv preprint arXiv:2502.19183},
  year={2025}
}
```

Paper: [https://arxiv.org/pdf/2502.19183](https://arxiv.org/pdf/2502.19183)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
