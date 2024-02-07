import enum
import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from autogluon.tabular import TabularDataset
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue.'

def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)

class Tokens():

    def __init__(
        self
    ) -> None:

        super().__init__()
        self.auto_features_to_tokens_generator = AutoMLPipelineFeatureGenerator()
        # self.num_tokens = self.auto_features_to_tokens_generator.features_out
        # self.num_features = self.auto_features_to_tokens_generator.features_in

    def fit(self, full_data: pd.DataFrame):
        self.auto_features_to_tokens_generator.fit_transform(X=full_data)
        self.num_tokens = len(self.auto_features_to_tokens_generator.features_out)
        self.num_features = len(self.auto_features_to_tokens_generator.features_in)
        
        return self.auto_features_to_tokens_generator.transform(X=full_data)
    
    def transform(self, target_data):
        return self.auto_features_to_tokens_generator.transform(X=target_data)

    def n_tokens(self):
        """The number of tokens."""
        try:
            num_tokens = self.num_tokens
            return num_tokens
        except Exception as e:
            print('Have not fit auto_features_to_tokens_generator object')
            print('发生以下异常：',e)
            exit()

    def n_feats(self):
        """The number of features."""
        try:
            num_features = self.num_features
            return num_features
        except Exception as e:
            print('Have not fit auto_features_to_tokens_generator object')
            print('发生以下异常：',e)
            exit()

class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) -> '_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)

    
class UnifiedFeatureTokenizer(nn.Module):
    """Transforms continuous features to tokens (embeddings).

    See `FeatureTokenizer` for the illustration.

    For one feature, the transformation consists of two steps:

    * the feature is multiplied by a trainable vector
    * another trainable vector is added

    Note that each feature has its separate pair of trainable vectors, i.e. the vectors
    are not shared between features.

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            d_token = 3
            tokenizer = UnifiedFeatureTokenizer(Tokens:pd.Dataframe, d_token, True, 'uniform')
            tokens = tokenizer(x)
            assert tokens.shape == (n_objects, n_features, d_token)
    """

    def __init__(
        self,
        Tokens: Tokens,
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Args:
            n_features: the number of continuous (scalar) features
            d_token: the size of one token
            bias: if `False`, then the transformation will include only multiplication.
                **Warning**: :code:`bias=False` leads to significantly worse results for
                Transformer-like (token-based) architectures.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
                In [gorishniy2021revisiting], the 'uniform' initialization was used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.Tokens = Tokens
        self.num_tokens = self.Tokens.n_tokens()
        self.num_features = self.Tokens.n_feats()
        self.weight = nn.Parameter(Tensor(self.num_tokens, d_token))
        self.bias = nn.Parameter(Tensor(self.num_tokens, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.weight.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class FeatureTokenizer(nn.Module):
    """Combines `NumericalFeatureTokenizer` and `CategoricalFeatureTokenizer`.

    The "Feature Tokenizer" module from [gorishniy2021revisiting]. The module transforms
    continuous and categorical features to tokens (embeddings).

    In the illustration below, the red module in the upper brackets represents
    `NumericalFeatureTokenizer` and the green module in the lower brackets represents
    `CategoricalFeatureTokenizer`.

    .. image:: ../images/feature_tokenizer.png
        :scale: 33%
        :alt: Feature Tokenizer

    Examples:
        .. testcode::

            n_objects = 4
            n_num_features = 3
            n_cat_features = 2
            d_token = 7
            x_num = torch.randn(n_objects, n_num_features)
            x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
            # [2, 3] reflects cardinalities fr
            tokenizer = FeatureTokenizer(n_num_features, [2, 3], d_token)
            tokens = tokenizer(x_num, x_cat)
            assert tokens.shape == (n_objects, n_num_features + n_cat_features, d_token)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    def __init__(
        self,
        # n_num_features: int,
        Tokens: Tokens,        
        d_token: int,
    ) -> None:
        """
        Args:
            n_num_features: the number of continuous features. Pass :code:`0` if there
                are no numerical features.
            cat_cardinalities: the number of unique values for each feature. See
                `CategoricalFeatureTokenizer` for details. Pass an empty list if there
                are no categorical features.
            d_token: the size of one token.
        """
        super().__init__()
        self.initialization = 'uniform'
        self.feature_tokenizer = UnifiedFeatureTokenizer(
                                                    Tokens=Tokens,
                                                    d_token=d_token,
                                                    bias=True,
                                                    initialization=self.initialization,
                                                    )
        self.cat_tokenizer = None

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return sum(
            x.n_tokens
            for x in [self.feature_tokenizer, self.cat_tokenizer]
            if x is not None
        )

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return (
            self.cat_tokenizer.d_token  # type: ignore
            if self.feature_tokenizer is None
            else self.feature_tokenizer.d_token
        )


    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        """Perform the forward pass.

        Args:
            x_num: continuous features. Must be presented if :code:`n_num_features > 0`
                was passed to the constructor.
            x_cat: categorical features (see `CategoricalFeatureTokenizer.forward` for
                details). Must be presented if non-empty :code:`cat_cardinalities` was
                passed to the constructor.
        Returns:
            tokens
        Raises:
            AssertionError: if the described requirements for the inputs are not met.
        """
        assert (
            x_num is not None or x_cat is not None
        ), 'At least one of x_num and x_cat must be presented'
        assert _all_or_none(
            [self.feature_tokenizer, x_num]
        ), 'If self.feature_tokenizer is (not) None, then x_num must (not) be None'
        assert _all_or_none(
            [self.cat_tokenizer, x_cat]
        ), 'If self.cat_tokenizer is (not) None, then x_cat must (not) be None'
        x = []
        if self.feature_tokenizer is not None:
            x.append(self.feature_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)

class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.

    To learn about the [CLS]-based inference, see [devlin2018bert].

    When used as a module, the [CLS]-token is appended **to the end** of each item in
    the batch.

    Examples:
        .. testcode::

            batch_size = 2
            n_tokens = 3
            d_token = 4
            cls_token = CLSToken(d_token, 'uniform')
            x = torch.randn(batch_size, n_tokens, d_token)
            x = cls_token(x)
            assert x.shape == (batch_size, n_tokens + 1, d_token)
            assert (x[:, -1, :] == cls_token.expand(len(x))).all()

    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    """

    def __init__(self, d_token: int, initialization: str) -> None:
        """
        Args:
            d_token: the size of token
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(d_token))
        initialization_.apply(self.weight, d_token)


    def expand(self, *leading_dimensions: int) -> Tensor:
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.

        A possible use case is building a batch of [CLS]-tokens. See `CLSToken` for
        examples of usage.

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the
            underlying :code:`weight` parameter, so gradients will be propagated as
            expected.

        Args:
            leading_dimensions: the additional new dimensions

        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)



    def forward(self, x: Tensor) -> Tensor:
        """Append self **to the end** of each item in the batch (see `CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)
    
if __name__ == '__main__':

    import torch
    # 创建数据集对象，使用pandas的Dataframe对象即可
    import pandas as pd
    import numpy as np
    import random
    from sklearn.datasets import make_regression
    from datetime import datetime    

    x, y = make_regression(n_samples = 100,n_features = 5,n_targets = 1, random_state = 1)
    dfx = pd.DataFrame(x, columns=['A','B','C','D','E'])
    dfy = pd.DataFrame(y, columns=['label'])

    # Create an integer column, a datetime column, a categorical column and a string column to demonstrate how they are processed.
    dfx['B'] = (dfx['B']).astype(int)
    dfx['C'] = datetime(2000,1,1) + pd.to_timedelta(dfx['C'].astype(int), unit='D')
    dfx['D'] = pd.cut(dfx['D'] * 10, [-np.inf,-5,0,5,np.inf],labels=['v','w','x','y'])
    dfx['E'] = pd.Series(list(' '.join(random.choice(["abc", "d", "ef", "ghi", "jkl"]) for i in range(4)) for j in range(100)))
    dataset=TabularDataset(dfx)    

    # Token之后需要对数据实例化
    feature_to_tokens = Tokens()
    dfx = feature_to_tokens.fit(full_data=dataset)
    dfx = dfx.dropna(subset=['E'])
    print('Sucessfully fitting feature_to_toekns_generator!')

    featurizer = FeatureTokenizer(Tokens=feature_to_tokens, d_token=128)
    print('Sucessfully initializing FeatureTokenizer!')

    # 注意featurizer需要对输入数据进行向量化
    ccc = featurizer(torch.tensor(dfx.iloc[:2].values),None)
    print(ccc)
    print(ccc.shape)

    clsToken = CLSToken(featurizer.d_token, featurizer.initialization)
    ccc_tokenizer = clsToken(ccc)
    print(ccc_tokenizer)
    print(ccc_tokenizer.shape)