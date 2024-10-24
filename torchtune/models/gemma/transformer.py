# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules.transformer import _get_clones, TransformerSelfAttentionLayer
from torchtune.utils._logging import deprecated


@deprecated(
    msg="Please use torchtune.modules.TransformerDecoder instead. \
If you need an example, see torchtune.models.gemma._component_builders.py"
)
class GemmaTransformerDecoder(nn.Module):
    """
    GemmaTransformer Decoder derived from Gemma architecture. A key difference between
    the Gemma transformer decoder and :class:`~torchtune.modules.TransformerDecoder`
    is that the output projection is replaced instead with a reverse projection
    using the transposed token embedding weights from output dim to input dim
    (see https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/layers/modeling/reversible_embedding.py#L21).

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space and as the output projection.
        layer (TransformerSelfAttentionLayer): Transformer Decoder layer.
        num_layers (int): Number of Transformer Decoder layers.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            :func:`~torchtune.modules.KVCache`
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the :func:`~torchtune.modules.KVCache`
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        norm_embeddings (bool): Whether to normalize the embeddings before passing them
            through the decoder layers. Defaults to False.

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        tok_embeddings: nn.Embedding,
        layer: TransformerSelfAttentionLayer,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        norm_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.tok_embeddings = tok_embeddings
        self.layers = _get_clones(layer, num_layers)
        self.norm = norm
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None
        self.norm_embeddings = norm_embeddings
        self.num_output_chunks = 0

    def caches_are_setup(self) -> bool:
        """Check if the key value caches are setup."""
        return self.layers[0].cache_enabled

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe."""
        self.num_output_chunks = num_output_chunks

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ):
        """
        Sets up key-value attention caches for inference. For each layer in ``self.layers``:
        - :class:`torchtune.modules.TransformerSelfAttentionLayer` will use ``decoder_max_seq_len``.
        - :class:`torchtune.modules.TransformerCrossAttentionLayer` will use ``encoder_max_seq_len``.
        - :class:`torchtune.modules.fusion.FusionLayer` will use both ``decoder_max_seq_len`` and ``encoder_max_seq_len``.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (Optional[int]): maximum encoder cache sequence length.
            decoder_max_seq_len (Optional[int]): maximum decoder cache sequence length.
        """
        if encoder_max_seq_len is not None:
            self.encoder_max_seq_len = encoder_max_seq_len
        if decoder_max_seq_len is not None:
            self.decoder_max_seq_len = decoder_max_seq_len
        for layer in self.layers:
            layer.setup_caches(
                batch_size,
                dtype,
                encoder_max_seq_len=encoder_max_seq_len,
                decoder_max_seq_len=decoder_max_seq_len,
            )

        # causal_mask is used during inference to ensure we're attending
        # to the right tokens
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool)
        )

    @torch.compiler.disable
    def chunked_output(self, last_hidden_state: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply output projection in chunks. This should be applied in conjunction with
        :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss` as upcasting to fp32 is done there.

        To use this method, you should first call
        :func:`~torchtune.models.gemma.GemmaTransformerDecoder.set_num_output_chunks`.

        Args:
            last_hidden_state (torch.Tensor): last hidden state of the decoder, having shape
                [b, seq_len, embed_dim].

        Returns:
            List[torch.Tensor]: List of num_chunks output tensors, each with shape
                [b, seq_len/num_chunks, out_dim], where out_dim is usually the vocab size.
        """
        return [
            F.linear(chunk, self.tok_embeddings.weight)
            for chunk in last_hidden_state.chunk(self.num_output_chunks, dim=1)
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape [b x s]
            mask (Optional[torch.Tensor]): Optional boolean tensor which contains the attention mask
                with shape [b x s x s]. This is applied after the query-key multiplication and
                before the softmax. A value of True in row i and column j means token i attends
                to token j. A value of False means token i does not attend to token j. If no
                mask is specified, a causal mask is used by default. Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token and
                is required.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` should contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            torch.Tensor: output tensor with shape [b x s x v]

        Raises:
            ValueError: if causal_mask is set but input_pos is None

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - v: vocab size
            - d: embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        if self.causal_mask is not None:
            if input_pos is None:
                raise ValueError(
                    "Caches are setup, but the position of input token is missing"
                )
            if mask is not None:
                raise ValueError(
                    "An attention mask was set. Cannot use a non-causal mask for inference"
                )
            # shape: [1, input_pos_len, m_s]
            # in most cases input_pos_len should be 1
            mask = self.causal_mask[None, input_pos]

        if self.norm_embeddings:
            hidden_dim = h.size(-1)
            h = h * torch.tensor(hidden_dim**0.5, dtype=h.dtype)

        for layer in self.layers:
            # shape: [b, s, d]
            h = layer(h, mask=mask, input_pos=input_pos)

        # shape: [b, s, d]
        h = self.norm(h)

        if self.num_output_chunks > 0:
            output = self.chunked_output(h)
        else:
            # shape: [b, seq_len, out_dim]
            output = F.linear(h, self.tok_embeddings.weight).float()
        return output
