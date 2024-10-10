# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
import os
import sys

sys.path.append("/data/users/ebs/llama3_2_vision")
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import torch
from llama3_2_vision._convert_weights import (  # TODO: convert to torchtune import
    llama3_vision_meta_to_tune,
)
from safetensors.torch import save_file
from torchtune import training

from torchtune.models import convert_weights
from torchtune.models.phi3._convert_weights import phi3_hf_to_tune, phi3_tune_to_hf
from torchtune.models.qwen2._convert_weights import qwen2_hf_to_tune, qwen2_tune_to_hf
from torchtune.rlhf.utils import reward_hf_to_tune, reward_tune_to_hf
from torchtune.training.checkpointing._utils import (
    get_path,
    # ModelType,
    safe_torch_load,
    save_config,
)
from torchtune.utils import get_logger

logger = get_logger("DEBUG")

from torchtune.training.checkpointing._checkpointer import _CheckpointerInterface


# TODO: Update ModelType to match below
class ModelType(Enum):
    """ModelType is used by the checkpointer to distinguish between different model architectures.

    If you are adding a new model that follows a different format than those in the repo already,
    you can add a new ModelType to gate on weight conversion logic unique to that model.

    Attributes:
        GEMMA (str): Gemma family of models. See :func:`~torchtune.models.gemma.gemma`
        LLAMA2 (str): Llama2 family of models. See :func:`~torchtune.models.llama2.llama2`
        LLAMA3 (str): Llama3 family of models. See :func:`~torchtune.models.llama3.llama3`
        LLAMA3_VISION (str): LLama3 vision family of models. See :func:`~torchtune.models.llama3_2_vision.llama3_vision_decoder`
        MISTRAL (str): Mistral family of models. See :func:`~torchtune.models.mistral.mistral`
        PHI3_MINI (str): Phi-3 family of models. See :func:`~torchtune.models.phi3.phi3`
        REWARD (str): A Llama2, Llama3, or Mistral model with a classification head projecting
            to a single class for reward modelling.
            See :func:`~torchtune.models.mistral.mistral_reward_7b` or :func:`~torchtune.models.llama2.llama2_reward_7b`
        QWEN2 (str): Qwen2 family of models. See :func:`~torchtune.models.qwen2.qwen2`

    Example:
        >>> # Usage in a checkpointer class
        >>> def load_checkpoint(self, ...):
        >>>     ...
        >>>     if self._model_type == MY_NEW_MODEL:
        >>>         state_dict = my_custom_state_dict_mapping(state_dict)
    """

    GEMMA: str = "gemma"
    LLAMA2: str = "llama2"
    LLAMA3: str = "llama3"
    LLAMA3_VISION: str = "llama3_vision"
    MISTRAL: str = "mistral"
    PHI3_MINI: str = "phi3_mini"
    REWARD: str = "reward"
    QWEN2: str = "qwen2"


# TODO: Update FullModelMetaCheckpointer to match below
class FullModelMetaCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in Meta's format. Examples include
    the Llama-2-7b model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b)

    Currently we support reading from a single checkpoint file only. Support for reading from
    sharded checkpoints is WIP.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (List[str]): List of checkpoint files to load. Currently this checkpointer only
            supports loading a single checkpoint file.
        model_type (ModelType): Model type of the model for which the checkpointer is being loaded
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. Default is None
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. Default is None
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files to
            resume training from a previous run. Default is False

    Raises:
        ValueError: If ``checkpoint_files`` is not a list of length 1
        ValueError: If ``resume_from_checkpoint`` is True but ``recipe_checkpoint`` is None
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: List[str],
        model_type: ModelType,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ) -> None:
        # Fail fast if ``checkpoint_files`` is invalid
        if len(checkpoint_files) != 1:
            raise ValueError(
                "Currently we only support reading from a single torchtune checkpoint file. "
                f"Got {len(checkpoint_files)} files instead."
            )

        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_path = get_path(self._checkpoint_dir, checkpoint_files[0])

        self._adapter_checkpoint = (
            get_path(self._checkpoint_dir, adapter_checkpoint)
            if adapter_checkpoint
            else None
        )

        self._resume_from_checkpoint = resume_from_checkpoint
        self._model_type = ModelType[model_type]
        self._output_dir = Path(output_dir)

        # recipe_checkpoint contains the recipe state. This should be available if
        # resume_from_checkpoint is True
        self._recipe_checkpoint = None
        if self._resume_from_checkpoint:
            if recipe_checkpoint is None:
                raise ValueError(
                    "If resume_from_checkpoint is True, recipe_checkpoint file must be provided."
                )
            self._recipe_checkpoint = get_path(self._checkpoint_dir, recipe_checkpoint)

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load Meta checkpoint from file. Currently only loading from a single file is supported.
        """
        state_dict: Dict[str:Any] = {}
        model_state_dict = safe_torch_load(self._checkpoint_path)
        if self._model_type == ModelType.LLAMA3_VISION:
            state_dict[training.MODEL_KEY] = llama3_vision_meta_to_tune(
                model_state_dict
            )
        else:
            state_dict[training.MODEL_KEY] = convert_weights.meta_to_tune(
                model_state_dict
            )

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._resume_from_checkpoint:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            state_dict.update(recipe_state)
        return state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
    ) -> None:
        """
        Save Meta checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir`` which contains the recipe
        state.

        Args:
            state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
            adapter_only (bool): If True, only save the adapter weights. Default is False

        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """
        self._output_dir.mkdir(exist_ok=True)

        if not adapter_only:
            model_state_dict = state_dict[training.MODEL_KEY]
            if self._model_type == ModelType.LLAMA3_VISION:
                # TODO add support for saving
                raise NotImplementedError(
                    "Saving checkpoints for Llama3 Vision is not supported yet."
                )
            else:
                state_dict[training.MODEL_KEY] = convert_weights.tune_to_meta(
                    model_state_dict
                )

            # Output file is always a .pt file with the epoch number in the name
            checkpoint_file = Path.joinpath(
                self._output_dir, f"meta_model_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[training.MODEL_KEY], checkpoint_file)
            logger.info(
                "Model checkpoint of size "
                f"{os.path.getsize(checkpoint_file) / 1000**3:.2f} GB "
                f"saved to {checkpoint_file}"
            )

        if training.ADAPTER_KEY in state_dict:
            output_path = Path.joinpath(
                self._output_dir, f"adapter_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[training.ADAPTER_KEY], output_path)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )
        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        # If the recipe state needs to be output, first remove the model state dict
        # and if it exists, remove the adapter state dict as well
        if intermediate_checkpoint:
            _ = state_dict.pop(training.MODEL_KEY)
            _ = state_dict.pop(training.ADAPTER_KEY, None)
            _ = state_dict.pop(training.ADAPTER_CONFIG, None)
            output_path = Path.joinpath(self._output_dir, "recipe_state.pt")
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )
        else:
            logger.info("Saving final epoch checkpoint.")
            if adapter_only:
                logger.info(
                    "Please note that you have set adapter_only=True, so only adapter weights will be saved."
                    "You need to merge the adapter weights into your base model for further use. "
                    f"See {self.__class__.__name__}.save_checkpoint for more details."
                )
            else:
                logger.info(
                    "The full model checkpoint, including all weights and configurations, has been saved successfully."
                    "You can now use this checkpoint for further training or inference."
                )
