"""
Refs:
    - https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.embed
    - https://huggingface.co/BAAI/bge-m3
"""

from typing import Dict, Union, List

import llama_cpp

import numpy as np
import torch

from schemas.embed import EmbedResult

from utils.logger import _logger

logger = _logger(__name__)


class LlamaCppSession(llama_cpp.Llama):

    def __init__(self, *args, **kwargs):
        kwargs["embedding"] = True
        kwargs["pooling_type"] = 0

        super().__init__(*args, **kwargs)

        sparse_state_dict = torch.load(
            "models/sparse_linear.pt", map_location="cpu", weights_only=True
        )
        self.sparse_linear = torch.nn.Linear(in_features=1024, out_features=1)
        self.sparse_linear.load_state_dict(sparse_state_dict)
        self.colbert_linear = torch.nn.Linear(
            in_features=1024, out_features=1024
        )

    def _process_token_weights(
        self, token_weights: np.ndarray, input_ids: List[int]
    ) -> Dict[int, float]:
        from functools import reduce

        if len(token_weights.shape) == 2:
            token_weights = token_weights.squeeze(-1)

        result = reduce(
            lambda acc, x: {
                **acc, x[0]: x[1]
            } if ((x[0] not in acc) or acc[x[0]] < x[1]) else {**acc},
            zip(input_ids, token_weights), {}
        )

        return result

    def compute_score(self, lw1: Dict[int, float], lw2: Dict[int, float]):
        scores = 0
        for token, weight in lw1.items():
            if token in lw2:
                scores += weight * lw2[token]
        return scores

    def _sparse_embedding(
        self, hidden_state: List[List[float]], input_ids: List[int]
    ):
        sparse_tensors = torch.tensor(hidden_state)
        norm = torch.nn.LayerNorm(sparse_tensors.size())
        sparse_tensors = norm(sparse_tensors)

        sparse_tensors = torch.nan_to_num(sparse_tensors, 0)

        token_weights_tensor = torch.relu(self.sparse_linear(sparse_tensors))
        input_ids_tensor = torch.tensor(input_ids)

        logger(f"Hidden state: {sparse_tensors.size()}")
        logger(f"Token weights: {token_weights_tensor.size()}")
        logger(f"Input ids: {input_ids_tensor.size()}")

        sparse_embedding_tensor = torch.zeros(
            input_ids_tensor.size(0),
            250002,
            dtype=token_weights_tensor.dtype,
            device=token_weights_tensor.device
        )
        sparse_embedding_tensor = torch.scatter(
            sparse_embedding_tensor,
            dim=1,
            index=input_ids_tensor.unsqueeze(-1),
            src=token_weights_tensor
        )
        logger(f"Sparse embedding tensor: {sparse_embedding_tensor.size()}")

        unused_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
        unused_token_ids = [
            self.tokenize(x.encode("utf-8"))[0] for x in unused_tokens
        ]

        sparse_result_tensor = torch.max(sparse_tensors, dim=0).values
        logger(f"Sparse result tensor: {sparse_result_tensor.size()}")
        sparse_result_tensor[unused_token_ids] *= 0.

        sparse_result_tensor = torch.nan_to_num(sparse_result_tensor, 0)
        token_weights_tensor = torch.nan_to_num(token_weights_tensor, 0)

        return sparse_result_tensor, token_weights_tensor

    def embed_(
        self,
        input: Union[str, List[str]],
        truncate: bool = True,
    ) -> List[EmbedResult]:
        """Embed a string with lexical weights

        Args:
            input: The utf-8 encoded string to embed.

        Returns:
            A list of embeddings
        """
        n_embd = self.n_embd()
        n_batch = self.n_batch

        # get pooling information
        pooling_type = self.pooling_type()
        logits_all = pooling_type == llama_cpp.LLAMA_POOLING_TYPE_NONE
        if self.context_params.embeddings is False:
            raise RuntimeError(
                "Llama model must be created with embedding=True to call this method"
            )

        if self.verbose:
            llama_cpp.llama_perf_context_reset(self._ctx.ctx)

        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input

        # reset batch
        self._batch.reset()

        # decode and fetch embeddings
        input_ids: List[List[int]] = []
        seq_embeddings: List[List[List[float]]] = []
        cls_embeddings: List[List[float]] = []

        def decode_batch(seq_sizes: List[int]):
            llama_cpp.llama_kv_cache_clear(self._ctx.ctx)
            self._ctx.decode(self._batch)
            self._batch.reset()

            pos: int = 0

            for size in seq_sizes:
                ptr = llama_cpp.llama_get_embeddings(self._ctx.ctx)

                embeddings = [
                    ptr[pos + j * n_embd:pos + (j + 1) * n_embd]
                    for j in range(size)
                ]

                seq_embeddings.append(embeddings)
                cls_embeddings.append(embeddings[0])

                pos += size

        # init state
        total_tokens = 0
        s_batch = []
        t_batch = 0
        p_batch = 0

        # accumulate batches and encode
        for text in inputs:
            _input_ids = self.tokenize(text.encode("utf-8"))
            if truncate:
                _input_ids = _input_ids[:n_batch]

            input_ids.append(_input_ids)

            n_tokens = len(_input_ids)
            total_tokens += n_tokens

            # check for overrun
            if n_tokens > n_batch:
                raise ValueError(
                    f"Requested tokens ({n_tokens}) exceed batch size of {n_batch}"
                )

            # time to eval batch
            if t_batch + n_tokens > n_batch:
                decode_batch(s_batch)
                s_batch = []
                t_batch = 0
                p_batch = 0

            # add to batch
            self._batch.add_sequence(_input_ids, p_batch, logits_all)

            # update batch stats
            s_batch.append(n_tokens)
            t_batch += n_tokens
            p_batch += 1

        # hanlde last batch
        decode_batch(s_batch)

        if self.verbose:
            llama_cpp.llama_perf_context_print(self._ctx.ctx)

        lexical_weights: List[Dict[int, float]] = []
        token_weights = []
        for idx, input_id in enumerate(input_ids):
            _, token_weight = self._sparse_embedding(
                seq_embeddings[idx], input_id
            )

            lexical_weights.append(
                self._process_token_weights(
                    token_weight.detach().numpy(), input_id
                )
            )
            token_weights.append(token_weight)

        outputs = [
            EmbedResult(
                dense=cls_embeddings[idx],
                sparse=lexical_weights[idx],
            ) for idx in range(len(input_ids))
        ]

        llama_cpp.llama_kv_cache_clear(self._ctx.ctx)
        self.reset()

        return outputs
