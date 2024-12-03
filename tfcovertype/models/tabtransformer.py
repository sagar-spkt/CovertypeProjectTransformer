import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
from transformers import PreTrainedModel

from ..config import TabularDataClassifierConfig


class TabTransformerClassifier(PreTrainedModel):
    """
    A classifier model based on the TabTransformer architecture for tabular data.

    This model inherits from `PreTrainedModel` and is designed to handle both
    categorical and continuous input features, producing classification logits.

    Parameters
    ----------
    config : TabularDataClassifierConfig
        Configuration object containing hyperparameters for the TabTransformer model,
        including information about category dimensions, continuous feature count,
        embedding dimensions, attention depth, and dropout rates.

    Attributes
    ----------
    model : TabTransformer
        An instance of the TabTransformer model used for tabular data classification.

    Methods
    -------
    forward(x_categ, x_cont, labels=None)
        Forward pass through the model, returning logits and optionally the loss.
    """
    config_class = TabularDataClassifierConfig

    def __init__(self, config):
        """
        Initializes the TabTransformerClassifier.

        Parameters
        ----------
        config : TabularDataClassifierConfig
            Configuration object with model hyperparameters.
        """
        super(TabTransformerClassifier, self).__init__(config)
        self.model = TabTransformer(
            categories=config.categories,
            num_continuous=config.num_continuous,
            dim=config.dim,
            dim_out=config.dim_out,
            depth=config.depth,
            heads=config.heads,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            mlp_hidden_mults=config.mlp_hidden_mults,
        )

    def forward(self, x_categ, x_cont, labels=None):
        """
        Forward pass through the TabTransformer model.

        Parameters
        ----------
        x_categ : torch.Tensor
            A tensor of shape (batch_size, num_categorical_features) representing
            the categorical input features. Each element should be an integer
            representing a category.

        x_cont : torch.Tensor
            A tensor of shape (batch_size, num_continuous_features) representing
            the continuous input features. Each element should be a float.

        labels : torch.Tensor, optional
            A tensor of shape (batch_size,) containing ground truth labels for
            classification. If provided, the cross-entropy loss is computed.

        Returns
        -------
        dict
            A dictionary containing:
            - 'logits' (torch.Tensor): The raw logits output of the model,
              of shape (batch_size, num_classes).
            - 'loss' (torch.Tensor, optional): The cross-entropy loss, returned
              if `labels` are provided.
        """
        logits = self.model(x_categ, x_cont)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return {'logits': logits, 'loss': loss}
        return {'logits': logits}
