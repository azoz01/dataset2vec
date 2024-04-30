from pydantic import BaseModel, Field
from pydantic.functional_validators import AfterValidator
from torch import nn
from typing import Annotated, Type

from dataset2vec.utils import all_elements_positive, non_empty, is_positive


class Dataset2VecConfig(BaseModel):
    """Configuration of the Dataset2Vec encoder"""

    activation_cls: Type[nn.Module] = Field(default=nn.ReLU)
    """Class of the activation function used in entire network."""
    f_dense_hidden_size: Annotated[int, AfterValidator(is_positive)] = 32
    """Size of the hidden layers of the first stage."""
    f_res_hidden_size: Annotated[int, AfterValidator(is_positive)] = 32
    """Size of the hidden layers of the residual blocks of the first stage."""
    f_res_n_layers: Annotated[int, AfterValidator(is_positive)] = 3
    """Number of the layers of the residual block of the first stage."""
    f_block_repetitions: Annotated[int, AfterValidator(is_positive)] = 7
    """Number of building blocks of the first stage."""
    f_out_size: Annotated[int, AfterValidator(is_positive)] = 32
    """Dimensionality of the output of the first starge."""
    g_layers_sizes: Annotated[
        list[int],
        AfterValidator(all_elements_positive),
        AfterValidator(non_empty),
    ] = [32, 16, 8]
    """Sizes of the layers of the feed forward network in the second stage."""
    h_dense_hidden_size: Annotated[int, AfterValidator(is_positive)] = 16
    """Size of the hidden layers of the feed forward net of the third stage."""
    h_res_hidden_size: Annotated[int, AfterValidator(is_positive)] = 16
    """Size of the hidden layers of the residual blocks of the third stage."""
    h_res_n_layers: Annotated[int, AfterValidator(is_positive)] = 3
    """Number of layers of the residual block of the third stage."""
    h_block_repetitions: Annotated[int, AfterValidator(is_positive)] = 3
    """Number of building blocks of the third stage."""
    output_size: Annotated[int, AfterValidator(is_positive)] = 16
    """Output dimensionality of the encoder."""
