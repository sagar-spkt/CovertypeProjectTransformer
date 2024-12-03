import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split


class CoverTypeDataset(Dataset):
    """
    A custom PyTorch Dataset for the CoverType classification task.

    This dataset handles numerical and categorical features, normalizes numerical features,
    and encodes categorical features for a neural network classifier.

    Attributes
    ----------
    NUMERICAL_FEATURES : list of str
        Names of the numerical features in the dataset.

    CATEGORICAL_FEATURES : list of str
        Names of the categorical features in the dataset.

    TARGET : str
        Name of the target column in the dataset.

    Methods
    -------
    __onehot2int(column)
        Converts one-hot encoded categorical features into single integer labels.

    __len__()
        Returns the number of samples in the dataset.

    __getitem__(index)
        Returns the categorical, continuous features, and label for a given index.

    __load_dataset()
        Loads the dataset from the HuggingFace dataset hub.

    get_dataset_splits(test_size=0.2, val_size=0.0, random_state=42)
        Returns training, validation, and test splits of the Covertype dataset.
    """

    NUMERICAL_FEATURES = [
        'elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology',
        'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways',
        'hillshade_9am', 'hillshade_noon', 'hillshade_3pm',
        'horizontal_distance_to_fire_points'
    ]
    CATEGORICAL_FEATURES = ['wilderness_area', 'soil_type']
    TARGET = 'cover_type'

    def __init__(self, dataframe, numerical_mean_std=None) -> None:
        """
        Initializes the CoverTypeDataset object.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input DataFrame containing tabular data.

        numerical_mean_std : tuple of torch.Tensor, optional
            A tuple containing the mean and standard deviation of the numerical features
            for normalization. If None, they are computed from the input DataFrame.
        """
        super(CoverTypeDataset, self).__init__()
        self.dataframe = dataframe

        # Convert one-hot encoded categorical features into integer labels
        self.dataframe = self.__onehot2int('wilderness_area')
        self.dataframe = self.__onehot2int('soil_type')

        # Extract categorical features and convert to torch tensor
        self.X_categ = torch.tensor(self.dataframe[self.CATEGORICAL_FEATURES].values)
        # Get the number of unique values for each categorical feature
        self.X_categ_unique_count = self.dataframe[self.CATEGORICAL_FEATURES].nunique().to_list()

        # Extract numerical features and convert to torch tensor
        self.X_cont = torch.tensor(self.dataframe[self.NUMERICAL_FEATURES].values)

        # Compute or use provided mean and standard deviation for normalization
        if numerical_mean_std is None:
            self.numerical_mean_std = self.X_cont.mean(axis=0), self.X_cont.std(axis=0)
            # self.categorical_unique_count = [self.dataframe[c].nunique() for c in self.CATEGORICAL_FEATURES]
        else:
            self.numerical_mean_std = numerical_mean_std

        # Normalize the numerical features
        self.X_cont = (self.X_cont - self.numerical_mean_std[0]) / self.numerical_mean_std[1]

        # Extract the unique class labels and convert the target to torch tensor
        self.classes = self.dataframe[self.TARGET].unique()
        self.y = torch.tensor(self.dataframe[self.TARGET].values)

    def __onehot2int(self, column):
        """
        Converts one-hot encoded columns to a single integer label.

        Parameters
        ----------
        column : str
            The prefix of the one-hot encoded columns.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the one-hot columns replaced by a single integer column.
        """
        selected_columns = [c for c in self.dataframe.columns if c.startswith(column)]
        # Find the index of the maximum value for each row and extract the integer label
        merged = self.dataframe[selected_columns].idxmax(axis=1).apply(lambda x: int(x.split('_')[-1]))
        # Drop the original one-hot columns and replace with the merged column
        df = self.dataframe.drop(columns=selected_columns)
        df[column] = merged
        return df

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples.
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Retrieves the features and label for a given index.

        Parameters
        ----------
        index : int
            The index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing:
            - 'x_categ' : torch.Tensor of shape (num_categorical_features,)
              Categorical feature values for the sample.
            - 'x_cont' : torch.Tensor of shape (num_numerical_features,)
              Numerical feature values for the sample.
            - 'labels' : torch.Tensor of shape (1,)
              The label for the sample.
        """
        return {"x_categ": self.X_categ[index], "x_cont": self.X_cont[index], "labels": self.y[index]}

    @staticmethod
    def __load_dataset():
        """
        Loads the CoverType dataset from the HuggingFace dataset hub.

        Returns
        -------
        pandas.DataFrame
            The dataset as a pandas DataFrame.
        """
        return load_dataset("mstz/covertype", "covertype")["train"].to_pandas()

    @classmethod
    def get_dataset_splits(cls, test_size=0.2, val_size=0.0, random_state=42):
        """
        Splits the dataset into training, validation, and test sets.

        Parameters
        ----------
        test_size : float, optional, default=0.2
            The proportion of the dataset to include in the test split.

        val_size : float, optional, default=0.0
            The proportion of the dataset to include in the validation split.

        random_state : int, optional, default=42
            Controls the randomness of the split.

        Returns
        -------
        tuple of CoverTypeDataset
            A tuple containing the training, validation (if specified), and test datasets.
        """
        # Load the full dataset as a DataFrame
        train_df = cls.__load_dataset()
        val_df, test_df = None, None

        # Split into training and test sets if test_size > 0
        if test_size > 0:
            train_df, test_df = train_test_split(
                train_df, test_size=test_size, random_state=random_state, stratify=train_df[cls.TARGET]
            )

        # Split into training and validation sets if val_size > 0
        if val_size > 0:
            train_df, val_df = train_test_split(
                train_df, test_size=val_size, random_state=random_state, stratify=train_df[cls.TARGET]
            )

        # Create dataset objects
        train_dataset, val_dataset, test_dataset = cls(train_df), None, None
        if test_df is not None:
            test_dataset = cls(test_df, train_dataset.numerical_mean_std)
        if val_df is not None:
            val_dataset = cls(val_df, train_dataset.numerical_mean_std)

        return train_dataset, val_dataset, test_dataset
