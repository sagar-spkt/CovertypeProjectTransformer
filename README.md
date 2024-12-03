# Transformer on Covertype Dataset

This project performs hyperparameter optimization (HPO) on the Covertype dataset using Transformer Models, specifically TabTransformer and FTTransformer. The repo includes scripts for training and visualizing the findings with Jupyter Notebook.

## Project Structure

```plaintext
├── tfcovertype                 # Main module
│   ├── __init__.py
│   ├── config.py               # TabTransformer and FTTransformer configuration class
│   ├── dataset.py              # Covertype dataset reading and preprocessing class
│   ├── utils.py                # Contains utils for model evaluation
│   └── models
│       ├── __init__.py
│       ├── fttransformer.py    # FTTransformer Implementation
│       └── tabtransformer.py   # TabTransformer Implementation
├── README.md                   # This file
├── requirements.txt            # List of dependencies
├── train.py                    # Script for running HPO
└── Report.ipynb                # Jupyter notebook for analysis and visualization
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Jupyter Notebook

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sagar-spkt/CovertypeProjectTransformer.git
    cd CovertypeProjectTransformer
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Training Script

The training script, `train.py`, allows you to perform hyperparameter optimization. You can configure the number of trials and specify the output directory for the results and other artifacts.

#### Example Usage:
```bash
python train.py --n-trials 20 --output_dir hpo_search
```

#### Arguments:
- `--n-trials`: Number of HPO trials to run (default: 10).
- `--output_dir`: Directory to save the results (default: `hpo_search`).

### Analyzing the Results

After running the training script, use the provided Jupyter Notebook `Report.ipynb` to analyze and visualize the results.

1. Open the notebook:
    ```bash
    jupyter notebook Report.ipynb
    ```

2. Follow the instructions in the notebook to generate plots and analyze the saved artifacts.

## Output

- HPO results will be saved in the specified output directory.
- The notebook will help in visualizing performance metrics and generating insights from the saved results.

## Support

If you encounter any issues, please open a ticket in the repository's issue tracker.
