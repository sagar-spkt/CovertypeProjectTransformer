from transformers import PretrainedConfig


class TabularDataClassifierConfig(PretrainedConfig):
    """
    Configuration class for a tabular data classifier model.

    This configuration class inherits from `PretrainedConfig` and contains the
    hyperparameters needed to initialize a tabular data classifier using a transformer
    architecture.

    Attributes
    ----------
    model_type : str
        The type of model, set to "tabular_data_classifier".

    Parameters
    ----------
    categories : tuple of int, optional, default=(10, 5, 6, 5, 8)
        A tuple containing the number of unique values for each categorical feature.
        The length of the tuple represents the number of categorical features.

    num_continuous : int, optional, default=10
        The number of continuous (numerical) input features.

    dim_out : int, optional, default=45
        The output dimension representing the number of classes for classification.

    dim : int, optional, default=32
        The dimensionality of the feature embeddings and transformer layers.

    depth : int, optional, default=6
        The number of transformer blocks (layers) in the model.

    heads : int, optional, default=8
        The number of attention heads in each transformer block.

    attn_dropout : float, optional, default=0.1
        The dropout rate applied after the attention mechanism.

    ff_dropout : float, optional, default=0.1
        The dropout rate applied after the feed-forward network.

    mlp_hidden_mults : tuple of int, optional, default=(4, 2)
        Multipliers for the hidden dimensions in the final MLP layer relative
        to the output dimension. Each value represents a hidden layer size.

    **kwargs : dict
        Additional keyword arguments passed to the `PretrainedConfig` constructor.
    """

    model_type = "tabular_data_classifier"

    def __init__(self,
                 categories=(10, 5, 6, 5, 8),  # tuple containing the number of unique values within each category
                 num_continuous=10,  # number of continuous values
                 dim_out=45,  # number of classes, could be anything
                 dim=32,  # dimension, paper set at 32
                 depth=6,  # depth, paper recommended 6
                 heads=8,  # heads, paper recommends 8
                 attn_dropout=0.1,  # post-attention dropout
                 ff_dropout=0.1,  # feed forward dropout
                 mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
                 **kwargs):
        """
        Initializes the TabularDataClassifierConfig.

        Parameters
        ----------
        categories : tuple of int, optional, default=(10, 5, 6, 5, 8)
            A tuple containing the number of unique values for each categorical feature.

        num_continuous : int, optional, default=10
            The number of continuous (numerical) input features.

        dim_out : int, optional, default=45
            The output dimension representing the number of classes.

        dim : int, optional, default=32
            The dimensionality of the feature embeddings and transformer layers.

        depth : int, optional, default=6
            The number of transformer blocks (layers).

        heads : int, optional, default=8
            The number of attention heads in each transformer block.

        attn_dropout : float, optional, default=0.1
            The dropout rate applied after the attention mechanism.

        ff_dropout : float, optional, default=0.1
            The dropout rate applied after the feed-forward network.

        mlp_hidden_mults : tuple of int, optional, default=(4, 2)
            Multipliers for the hidden dimensions in the final MLP layer.

        **kwargs : dict
            Additional keyword arguments passed to the `PretrainedConfig` constructor.
        """
        self.categories = categories
        self.num_continuous = num_continuous
        self.dim_out = dim_out
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.mlp_hidden_mults = mlp_hidden_mults
        super(TabularDataClassifierConfig, self).__init__(**kwargs)
