import os
import argparse
import pandas as pd
from sklearn.model_selection import ParameterSampler
from torch.utils.data import ConcatDataset
from transformers import Trainer, TrainingArguments

from tfcovertype.dataset import CoverTypeDataset
from tfcovertype.config import TabularDataClassifierConfig
from tfcovertype.models.fttransformer import FTTransformerClassifier
from tfcovertype.models.tabtransformer import TabTransformerClassifier
from tfcovertype.utils import compute_metrics


def model_init(dataset, params):
    """
    Initialize the model based on the given parameters and dataset.

    Parameters
    ----------
    dataset : CoverTypeDataset
        The dataset used for model initialization.
    params : dict
        Dictionary containing hyperparameters for model initialization.

    Returns
    -------
    PreTrainedModel
        An instance of either FTTransformerClassifier or TabTransformerClassifier.
    """
    # Create a configuration object for the model based on dataset properties and provided hyperparameters
    config = TabularDataClassifierConfig(
        categories=dataset.X_categ_unique_count,  # Number of unique categories per categorical feature
        num_continuous=len(dataset.NUMERICAL_FEATURES),  # Number of continuous features
        dim_out=len(dataset.classes),  # Number of output classes
        dim=params["dim"],  # Embedding dimension
        depth=params["depth"],  # Depth of the transformer
        heads=params["heads"],  # Number of attention heads
        attn_dropout=params["attn_dropout"],  # Attention dropout rate
        ff_dropout=params["ff_dropout"],  # Feedforward dropout rate
    )

    # Return the appropriate model based on the 'model' parameter
    if params["model"] == "TabTransformer":
        return TabTransformerClassifier(config)
    return FTTransformerClassifier(config)


def hp_space(n_trials=10):
    """
    Define the hyperparameter search space and generate samples for hyperparameter optimization.

    Parameters
    ----------
    n_trials : int, optional
        Number of random samples to generate from the hyperparameter space (default is 10).

    Returns
    -------
    list of dict
        List of dictionaries, each containing a sampled set of hyperparameters.
    """
    param_dist = {
        "learning_rate": [1e-5, 1e-4, 1e-3],
        "per_device_train_batch_size": [32, 64, 128, 256, 512, 1024],
        "weight_decay": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "model": ["FTTransformer", "TabTransformer"],
        "dim": [8, 16, 32, 64, 128],
        "depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "heads": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "attn_dropout": [1e-5, 1e-3, 1e-2, 1e-1],
        "ff_dropout": [1e-5, 1e-3, 1e-2, 1e-1],
    }

    # Generate random samples of hyperparameters
    sampled_params = list(ParameterSampler(param_dist, n_iter=n_trials, random_state=42))

    # Add default parameters for both models to ensure they are included in the search
    sampled_params.extend([
        {
            "learning_rate": 1e-3,
            "per_device_train_batch_size": 128,
            "weight_decay": 0.01,
            "model": "FTTransformer",
            "dim": 32,
            "depth": 6,
            "heads": 8,
            "attn_dropout": 0.1,
            "ff_dropout": 0.1,
        },
        {
            "learning_rate": 1e-3,
            "per_device_train_batch_size": 128,
            "weight_decay": 0.01,
            "model": "TabTransformer",
            "dim": 32,
            "depth": 6,
            "heads": 8,
            "attn_dropout": 0.1,
            "ff_dropout": 0.1,
        }
    ])

    return sampled_params


def run_hpo(args):
    """
    Run hyperparameter optimization (HPO) and model training.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing the number of trials and output directory.
    """
    # Load and split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = CoverTypeDataset.get_dataset_splits(
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    )

    trial_results = []
    for params in hp_space(n_trials=args.n_trials):
        # Define training arguments for each trial
        training_args = TrainingArguments(
            output_dir="tfcovertype",  # Directory to save model outputs
            report_to="none",  # No reporting to external services
            eval_strategy="no",  # No evaluation during training
            save_strategy="no",  # Do not save checkpoints during hyperparameter search
            logging_strategy="epoch",  # Log metrics at the end of each epoch
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            num_train_epochs=20,  # Number of training epochs
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )

        # Initialize the trainer with the current hyperparameters
        trainer = Trainer(
            model=model_init(train_dataset, params),
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,  # Function to compute evaluation metrics
        )

        # Train the model
        trainer.train()

        # Evaluate the model on the validation dataset and store the results
        eval_results = trainer.evaluate(val_dataset)
        eval_results.update(params)
        trial_results.append(eval_results)

    # Save all trial results to a CSV file
    trial_results = pd.DataFrame(trial_results)
    os.makedirs(args.output_dir, exist_ok=True)
    trial_results.to_csv(f"{args.output_dir}/trial_results.csv", index=False)

    # Identify the best hyperparameter configuration based on validation accuracy
    best_param = trial_results.loc[trial_results['eval_accuracy'].idxmax()].to_dict()

    # Define final training arguments using the best hyperparameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="epoch",
        learning_rate=best_param["learning_rate"],
        weight_decay=best_param["weight_decay"],
        per_device_train_batch_size=best_param["per_device_train_batch_size"],
        num_train_epochs=50,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    # Train the final model on the combined training and validation datasets
    trainer = Trainer(
        model=model_init(train_dataset, best_param),
        args=training_args,
        train_dataset=ConcatDataset([train_dataset, val_dataset]),  # Combine train and validation split for retraining
        eval_dataset=test_dataset,  # Evaluate on the test set
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Evaluate the final model on the test dataset and print the results
    eval_results = trainer.evaluate()
    print(eval_results)


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run Hyperparameter Search on Covertype dataset.")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of HPO trials.")
    parser.add_argument("--output_dir", type=str, default="hpo_search", help="Output directory for results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_hpo(args)
