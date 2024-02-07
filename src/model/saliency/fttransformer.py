import enum
import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
import copy
import src.model.saliency.functional as rtdlF
from autogluon.tabular import TabularDataset
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from src.model.saliency.featurizer import FeatureTokenizer,Tokens, _TokenInitialization,CLSToken

ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue.'


def _is_glu_activation(activation: ModuleType):
    return (
        isinstance(activation, str)
        and activation.endswith('GLU')
        or activation in [ReGLU, GEGLU]
    )


def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)



class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return rtdlF.reglu(x)




class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return rtdlF.geglu(x)


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == 'ReGLU':
            return ReGLU()
        elif module_type == 'GEGLU':
            return GEGLU()
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(
                    f'Failed to construct the module {module_type} with the arguments {args}'
                ) from err
            return cls(*args)
    else:
        return module_type(*args)


class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-) with optional 'linear' attention.

    To learn more about Multihead Attention, see [devlin2018bert]. See the implementation
    of `Transformer` and the examples below to learn how to use the compression technique
    from [wang2020linformer] to speed up the module when the number of tokens is large.

    Examples:
        .. testcode::

            n_objects, n_tokens, d_token = 2, 3, 12
            n_heads = 6
            a = torch.randn(n_objects, n_tokens, d_token)
            b = torch.randn(n_objects, n_tokens * 2, d_token)
            module = MultiheadAttention(
                d_token=d_token, n_heads=n_heads, dropout=0.2, bias=True, initialization='kaiming'
            )

            # self-attention
            x, attention_stats = module(a, a, None, None)
            assert x.shape == a.shape
            assert attention_stats['attention_probs'].shape == (n_objects * n_heads, n_tokens, n_tokens)
            assert attention_stats['attention_logits'].shape == (n_objects * n_heads, n_tokens, n_tokens)

            # cross-attention
            assert module(a, b, None, None)

            # Linformer self-attention with the 'headwise' sharing policy
            k_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
            v_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
            assert module(a, a, k_compression, v_compression)

            # Linformer self-attention with the 'key-value' sharing policy
            kv_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
            assert module(a, a, kv_compression, kv_compression)

    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
        * [wang2020linformer] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma "Linformer: Self-Attention with Linear Complexity", 2020
    """

    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Args:
            d_token: the token size. Must be a multiple of :code:`n_heads`.
            n_heads: the number of heads. If greater than 1, then the module will have
                an addition output layer (so called "mixing" layer).
            dropout: dropout rate for the attention map. The dropout is applied to
                *probabilities* and do not affect logits.
            bias: if `True`, then input (and output, if presented) layers also have bias.
                `True` is a reasonable default choice.
            initialization: initialization for input projection layers. Must be one of
                :code:`['kaiming', 'xavier']`. `kaiming` is a reasonable default choice.
        Raises:
            AssertionError: if requirements for the inputs are not met.
        """
        super().__init__()
        if n_heads > 1:
            assert d_token % n_heads == 0, 'd_token must be a multiple of n_heads'
        assert initialization in ['kaiming', 'xavier']

        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)
        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if initialization == 'xavier' and (
                m is not self.W_v or self.W_out is not None
            ):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )


    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: Optional[nn.Linear],
        value_compression: Optional[nn.Linear],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Perform the forward pass.

        Args:
            x_q: query tokens
            x_kv: key-value tokens
            key_compression: Linformer-style compression for keys
            value_compression: Linformer-style compression for values
        Returns:
            (tokens, attention_stats)
        """
        assert _all_or_none(
            [key_compression, value_compression]
        ), 'If key_compression is (not) None, then value_compression must (not) be None'
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, _INTERNAL_ERROR_MESSAGE
        if key_compression is not None:
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)  # type: ignore

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, {
            'attention_logits': attention_logits,
            'attention_probs': attention_probs,
        }


class Transformer(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {'first_prenormalization': True, 'prenormalization': True}


    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation(activation) else 1),
                bias_first,
            )
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x



    class Head(nn.Module):
        """The final module of the `Transformer` that performs BERT-like inference."""

        def __init__(
            self,
            *,
            d_in: int,
            bias: bool,
            activation: ModuleType,
            normalization: ModuleType,
            d_out: int,
        ):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            x = x[:, -1]
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        last_layer_query_idx: Union[None, List[int], slice],
        n_tokens: Optional[int],
        kv_compression_ratio: Optional[float],
        kv_compression_sharing: Optional[str],
        head_activation: ModuleType,
        head_normalization: ModuleType,
        d_out: int,
    ) -> None:
        super().__init__()
        if isinstance(last_layer_query_idx, int):
            raise ValueError(
                'last_layer_query_idx must be None, list[int] or slice. '
                f'Do you mean last_layer_query_idx=[{last_layer_query_idx}] ?'
            )
        if not prenormalization:
            assert (
                not first_prenormalization
            ), 'If `prenormalization` is False, then `first_prenormalization` must be False'
        assert _all_or_none([n_tokens, kv_compression_ratio, kv_compression_sharing]), (
            'If any of the following arguments is (not) None, then all of them must (not) be None: '
            'n_tokens, kv_compression_ratio, kv_compression_sharing'
        )
        assert kv_compression_sharing in [None, 'headwise', 'key-value', 'layerwise']
        if not prenormalization:
            if self.WARNINGS['prenormalization']:
                warnings.warn(
                    'prenormalization is set to False. Are you sure about this? '
                    'The training can become less stable. '
                    'You can turn off this warning by tweaking the '
                    'rtdl.Transformer.WARNINGS dictionary.',
                    UserWarning,
                )
            assert (
                not first_prenormalization
            ), 'If prenormalization is False, then first_prenormalization is ignored and must be set to False'
        if (
            prenormalization
            and first_prenormalization
            and self.WARNINGS['first_prenormalization']
        ):
            warnings.warn(
                'first_prenormalization is set to True. Are you sure about this? '
                'For example, the vanilla FTTransformer with '
                'first_prenormalization=True performs SIGNIFICANTLY worse. '
                'You can turn off this warning by tweaking the '
                'rtdl.Transformer.WARNINGS dictionary.',
                UserWarning,
            )
            time.sleep(3)

        def make_kv_compression():
            assert (
                n_tokens and kv_compression_ratio
            ), _INTERNAL_ERROR_MESSAGE  # for mypy
            # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L83
            return nn.Linear(n_tokens, int(n_tokens * kv_compression_ratio), bias=False)

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression_ratio and kv_compression_sharing == 'layerwise'
            else None
        )

        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx

        self.blocks = nn.ModuleList([])
        for layer_idx in range(n_blocks):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token=d_token,
                        n_heads=attention_n_heads,
                        dropout=attention_dropout,
                        bias=True,
                        initialization=attention_initialization,
                    ),
                    'ffn': Transformer.FFN(
                        d_token=d_token,
                        d_hidden=ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                        activation=ffn_activation,
                    ),
                    'attention_residual_dropout': nn.Dropout(residual_dropout),
                    'ffn_residual_dropout': nn.Dropout(residual_dropout),
                    'output': nn.Identity(),  # for hooks-based introspection
                }
            )
            if layer_idx or not prenormalization or first_prenormalization:
                layer['attention_normalization'] = _make_nn_module(
                    attention_normalization, d_token
                )
            layer['ffn_normalization'] = _make_nn_module(ffn_normalization, d_token)
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert (
                        kv_compression_sharing == 'key-value'
                    ), _INTERNAL_ERROR_MESSAGE
            self.blocks.append(layer)

        self.head = Transformer.Head(
            d_in=d_token,
            d_out=d_out,
            bias=True,
            activation=head_activation,  # type: ignore
            normalization=head_normalization if prenormalization else 'Identity',
        )

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, layer, stage, x):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        if self.prenormalization:
            norm_key = f'{stage}_normalization'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer, stage, x, x_residual):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = layer[f'{stage}_residual_dropout'](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'{stage}_normalization'](x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.ndim == 3
        ), 'The input must have 3 dimensions: (n_objects, n_tokens, d_token)'
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)

            query_idx = (
                self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None
            )
            x_residual = self._start_residual(layer, 'attention', x)
            x_residual, _ = layer['attention'](
                x_residual if query_idx is None else x_residual[:, query_idx],
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if query_idx is not None:
                x = x[:, query_idx]
            x = self._end_residual(layer, 'attention', x, x_residual)

            x_residual = self._start_residual(layer, 'ffn', x)
            x_residual = layer['ffn'](x_residual)
            x = self._end_residual(layer, 'ffn', x, x_residual)
            x = layer['output'](x)

        x = self.head(x)
        return x


class FTTransformer(nn.Module):
    """The FT-Transformer model proposed in [gorishniy2021revisiting].

    Transforms features to tokens with `FeatureTokenizer` and applies `Transformer` [vaswani2017attention]
    to the tokens. The following illustration provides a high-level overview of the
    architecture:

    .. image:: ../images/ft_transformer.png
        :scale: 25%
        :alt: FT-Transformer

    The following illustration demonstrates one Transformer block for :code:`prenormalization=True`:

    .. image:: ../images/transformer_block.png
        :scale: 25%
        :alt: PreNorm Transformer block

    Examples:
        .. testcode::

            x_num = torch.randn(4, 3)
            x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])

            module = FTTransformer.make_baseline(
                n_num_features=3,
                cat_cardinalities=[2, 3],
                d_token=8,
                n_blocks=2,
                attention_dropout=0.2,
                ffn_d_hidden=6,
                ffn_dropout=0.2,
                residual_dropout=0.0,
                d_out=1,
            )
            x = module(x_num, x_cat)
            assert x.shape == (4, 1)

            module = FTTransformer.make_default(
                n_num_features=3,
                cat_cardinalities=[2, 3],
                d_out=1,
            )
            x = module(x_num, x_cat)
            assert x.shape == (4, 1)

        To learn more about the baseline and default parameters:

        .. testcode::

            baseline_parameters = FTTransformer.get_baseline_transformer_subconfig()
            default_parameters = FTTransformer.get_default_transformer_config()

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        * [vaswani2017attention] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need", 2017
    """

    def __init__(
        self, feature_tokenizer: FeatureTokenizer, transformer: Transformer
    ) -> None:
        """
        Note:
            `make_baseline` and `make_default` are the recommended constructors.
        """
        super().__init__()
        if transformer.prenormalization:
            assert 'attention_normalization' not in transformer.blocks[0], (
                'In the prenormalization setting, FT-Transformer does not '
                'allow using the first normalization layer '
                'in the first transformer block'
            )
        self.feature_tokenizer = feature_tokenizer
        self.cls_token = CLSToken(
            feature_tokenizer.d_token, feature_tokenizer.initialization
        )
        self.transformer = transformer


    @classmethod
    def get_baseline_transformer_subconfig(
        cls: Type['FTTransformer'],
    ) -> Dict[str, Any]:
        """Get the baseline subset of parameters for the backbone."""
        return {
            'attention_n_heads': 8,
            'attention_initialization': 'kaiming',
            'ffn_activation': 'ReGLU',
            'attention_normalization': 'LayerNorm',
            'ffn_normalization': 'LayerNorm',
            'prenormalization': True,
            'first_prenormalization': False,
            'last_layer_query_idx': None,
            'n_tokens': None,
            'kv_compression_ratio': None,
            'kv_compression_sharing': None,
            'head_activation': 'ReLU',
            'head_normalization': 'LayerNorm',
        }

    @classmethod
    def get_default_transformer_config(
        cls: Type['FTTransformer'], *, n_blocks: int = 3
    ) -> Dict[str, Any]:
        """Get the default parameters for the backbone.

        Note:
            The configurations are different for different values of:code:`n_blocks`.
        """
        assert 1 <= n_blocks <= 6
        grid = {
            'd_token': [96, 128, 192, 256, 320, 384],
            'attention_dropout': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
            'ffn_dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
        }
        arch_subconfig = {k: v[n_blocks - 1] for k, v in grid.items()}  # type: ignore
        baseline_subconfig = cls.get_baseline_transformer_subconfig()
        # (4 / 3) for ReGLU/GEGLU activations results in almost the same parameter count
        # as (2.0) for element-wise activations (e.g. ReLU or GELU; see the "else" branch)
        ffn_d_hidden_factor = (
            (4 / 3) if _is_glu_activation(baseline_subconfig['ffn_activation']) else 2.0
        )
        return {
            'n_blocks': n_blocks,
            'residual_dropout': 0.0,
            'ffn_d_hidden': int(arch_subconfig['d_token'] * ffn_d_hidden_factor),
            **arch_subconfig,
            **baseline_subconfig,
        }


    @classmethod
    def _make(
        cls,
        feature_tokenizer,
        transformer_config,
    ):

        if transformer_config['d_out'] is None:
            transformer_config['head_activation'] = None
        if transformer_config['kv_compression_ratio'] is not None:
            transformer_config['n_tokens'] = feature_tokenizer.n_tokens + 1
        return FTTransformer(
            feature_tokenizer,
            Transformer(**transformer_config),
        )


    @classmethod
    def make_baseline(
        cls: Type['FTTransformer'],
        *,
        featurizer: FeatureTokenizer,
        cat_cardinalities: Optional[List[int]],
        d_token: int,
        n_blocks: int,
        attention_dropout: float,
        ffn_d_hidden: int,
        ffn_dropout: float,
        residual_dropout: float,
        last_layer_query_idx: Union[None, List[int], slice] = None,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        d_out: int,
    ) -> 'FTTransformer':
        """Create a "baseline" `FTTransformer`.

        This variation of FT-Transformer was used in [gorishniy2021revisiting]. See
        `get_baseline_transformer_subconfig` to learn the values of other parameters.
        See `FTTransformer` for usage examples.

        Tip:
            `get_default_transformer_config` can serve as a starting point for choosing
            hyperparameter values.

        Args:
            n_num_features: the number of continuous features
            cat_cardinalities: the cardinalities of categorical features (see
                `CategoricalFeatureTokenizer` to learn more about cardinalities)
            d_token: the token size for each feature. Must be a multiple of :code:`n_heads=8`.
            n_blocks: the number of Transformer blocks
            attention_dropout: the dropout for attention blocks (see `MultiheadAttention`).
                Usually, positive values work better (even when the number of features is low).
            ffn_d_hidden: the *input* size for the *second* linear layer in `Transformer.FFN`.
                Note that it can be different from the output size of the first linear
                layer, since activations such as ReGLU or GEGLU change the size of input.
                For example, if :code:`ffn_d_hidden=10` and the activation is ReGLU (which
                is always true for the baseline and default configurations), then the
                output size of the first linear layer will be set to :code:`20`.
            ffn_dropout: the dropout rate after the first linear layer in `Transformer.FFN`.
            residual_dropout: the dropout rate for the output of each residual branch of
                all Transformer blocks.
            last_layer_query_idx: indices of tokens that should be processed by the last
                Transformer block. Note that for most cases there is no need to apply
                the last Transformer block to anything except for the [CLS]-token. Hence,
                runtime and memory can be saved by setting :code:`last_layer_query_idx=[-1]`,
                since the :code:`-1` is the position of [CLS]-token in FT-Transformer.
                Note that this will not affect the result in any way.
            kv_compression_ratio: apply the technique from [wang2020linformer] to speed
                up attention modules when the number of features is large. Can actually
                slow things down if the number of features is too low. Note that this
                option can affect task metrics in unpredictable way. Overall, use this
                option with caution. See `MultiheadAttention` for some examples and the
                implementation of `Transformer` to see how this option is used.
            kv_compression_sharing: weight sharing policy for :code:`kv_compression_ratio`.
                Must be one of :code:`[None, 'headwise', 'key-value', 'layerwise']`.
                See [wang2020linformer] to learn more about sharing policies.
                :code:`headwise` and :code:`key-value` are reasonable default choices. If
                :code:`kv_compression_ratio` is `None`, then this parameter also must be
                `None`. Otherwise, it must not be `None` (compression parameters must be
                shared in some way).
            d_out: the output size.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
            * [wang2020linformer] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma "Linformer: Self-Attention with Linear Complexity", 2020
        """
        transformer_config = cls.get_baseline_transformer_subconfig()
        for arg_name in [
            'n_blocks',
            'd_token',
            'attention_dropout',
            'ffn_d_hidden',
            'ffn_dropout',
            'residual_dropout',
            'last_layer_query_idx',
            'kv_compression_ratio',
            'kv_compression_sharing',
            'd_out',
        ]:
            transformer_config[arg_name] = locals()[arg_name]
        return cls._make(featurizer, transformer_config)



    @classmethod
    def make_default(
        cls: Type['FTTransformer'],
        *,
        n_num_features: int,
        cat_cardinalities: Optional[List[int]],
        n_blocks: int = 3,
        last_layer_query_idx: Union[None, List[int], slice] = None,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        d_out: int,
    ) -> 'FTTransformer':
        """Create the default `FTTransformer`.

        With :code:`n_blocks=3` (default) it is the FT-Transformer variation that is
        referred to as "default FT-Transformer" in [gorishniy2021revisiting]. See
        `FTTransformer` for usage examples. See `FTTransformer.make_baseline` for
        parameter descriptions.

        Note:
            The second component of the default FT-Transformer is the default optimizer,
            which can be created with the `make_default_optimizer` method.

        Note:
            According to [gorishniy2021revisiting], the default FT-Transformer is
            effective in the ensembling mode (i.e. when predictions of several default
            FT-Transformers are averaged). For a single FT-Transformer, it is still
            possible to achieve better results by tuning hyperparameter for the
            `make_baseline` constructor.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        transformer_config = cls.get_default_transformer_config(n_blocks=n_blocks)
        for arg_name in [
            'last_layer_query_idx',
            'kv_compression_ratio',
            'kv_compression_sharing',
            'd_out',
        ]:
            transformer_config[arg_name] = locals()[arg_name]
        return cls._make(n_num_features, cat_cardinalities, transformer_config)



    def optimization_param_groups(self) -> List[Dict[str, Any]]:
        """The replacement for :code:`.parameters()` when creating optimizers.

        Example::

            optimizer = AdamW(
                model.optimization_param_groups(), lr=1e-4, weight_decay=1e-5
            )
        """
        no_wd_names = ['feature_tokenizer', 'normalization', '.bias']
        assert isinstance(
            getattr(self, no_wd_names[0], None), FeatureTokenizer
        ), _INTERNAL_ERROR_MESSAGE
        assert (
            sum(1 for name, _ in self.named_modules() if no_wd_names[1] in name)
            == len(self.transformer.blocks) * 2
            - int('attention_normalization' not in self.transformer.blocks[0])  # type: ignore
            + 1
        ), _INTERNAL_ERROR_MESSAGE

        def needs_wd(name):
            return all(x not in name for x in no_wd_names)

        return [
            {'params': [v for k, v in self.named_parameters() if needs_wd(k)]},
            {
                'params': [v for k, v in self.named_parameters() if not needs_wd(k)],
                'weight_decay': 0.0,
            },
        ]



    def make_default_optimizer(self) -> torch.optim.AdamW:
        """Make the optimizer for the default FT-Transformer."""
        return torch.optim.AdamW(
            self.optimization_param_groups(),
            lr=1e-4,
            weight_decay=1e-5,
        )


    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        x = self.feature_tokenizer(x_num, x_cat)
        x = self.cls_token(x)
        x,x_embeddings = self.transformer(x)
        return x,x_embeddings

from fastargs.decorators import param

@param('model.saliency.transformer.token_hid_dim')
@param('model.saliency.transformer.num_blocks')
def get_model(feature_to_tokens, token_hid_dim, num_blocks):
    featurizer = FeatureTokenizer(Tokens=feature_to_tokens, d_token=token_hid_dim)
    return FTTransformer.make_baseline(
                                        featurizer=featurizer,
                                        cat_cardinalities=None,
                                        d_token=token_hid_dim,
                                        n_blocks=num_blocks,
                                        attention_dropout=0.2,
                                        ffn_d_hidden=128,
                                        ffn_dropout=0.2,
                                        residual_dropout=0.0,
                                        d_out=token_hid_dim,
                                        ) 