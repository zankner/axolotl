"""Module for models and model loading"""
import logging
import math
import os
from typing import Optional, Tuple  # noqa: F401

import bitsandbytes as bnb
import torch
import transformers
from optimum.bettertransformer import BetterTransformer
from peft import PeftConfig, prepare_model_for_kbit_training
from peft.tuners.lora import QuantLinear
from transformers import (  # noqa: F401
    AddedToken,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    LlamaConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from axolotl.prompt_tokenizers import LLAMA_DEFAULT_EOS_TOKEN
from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.dict import DictDefault
from axolotl.monkeypatch.hydra_utils import (
    replace_compute_loss,
    add_hydra_heads,
    replace_create_optimizer,
)

LOG = logging.getLogger("axolotl")


def load_model_config(cfg):
    model_config_name = cfg.base_model_config or cfg.base_model
    trust_remote_code = cfg.trust_remote_code is True
    return AutoConfig.from_pretrained(
        model_config_name, trust_remote_code=trust_remote_code
    )


def load_tokenizer(cfg):
    tokenizer_kwargs = {}
    use_fast = True  # this is the default

    if cfg.tokenizer_use_fast is not None:
        use_fast = cfg.tokenizer_use_fast
    if cfg.tokenizer_legacy is not None:
        # True is the default w/ https://github.com/huggingface/transformers/pull/25224
        tokenizer_kwargs["legacy"] = cfg.tokenizer_legacy

    tokenizer_cls = AutoTokenizer
    if cfg.tokenizer_type:
        tokenizer_cls = getattr(transformers, cfg.tokenizer_type)

    tokenizer_config = cfg.tokenizer_config or cfg.base_model_config
    tokenizer = tokenizer_cls.from_pretrained(
        tokenizer_config,
        trust_remote_code=cfg.trust_remote_code or False,
        use_fast=use_fast,
        **tokenizer_kwargs,
    )

    if (
        tokenizer.__class__.__name__
        in [
            "LlamaTokenizer",
            "LlamaTokenizerFast",
            "CodeLlamaTokenizer",
        ]
        and hasattr(tokenizer, "pad_token")
        and not tokenizer.pad_token
    ):
        # set a pad_token, but use eos_token so we don't add a new token
        tokenizer.pad_token = LLAMA_DEFAULT_EOS_TOKEN

    LOG.debug(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
    LOG.debug(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
    LOG.debug(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
    LOG.debug(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

    if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Mistral's official FA implementation requires left padding
    if cfg.is_mistral_derived_model and cfg.flash_attention and not cfg.sample_packing:
        tokenizer.padding_side = "left"

    if cfg.special_tokens:
        for k, val in cfg.special_tokens.items():
            tokenizer.add_special_tokens(
                {k: AddedToken(val, rstrip=False, lstrip=False, normalized=False)}
            )
    if cfg.tokens:
        tokenizer.add_tokens(
            [
                AddedToken(token, rstrip=False, lstrip=False, normalized=False)
                for token in cfg.tokens
            ]
        )

    return tokenizer


def load_model(
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizerBase,
    inference: bool = False,
) -> Tuple[PreTrainedModel, Optional[PeftConfig]]:
    """
    Load a model for a given configuration and tokenizer.
    """
    base_model = cfg.base_model
    base_model_config = cfg.base_model_config
    model_type = cfg.model_type
    model_config = load_model_config(cfg)

    # TODO refactor as a kwarg
    load_in_8bit = cfg.load_in_8bit

    if hasattr(model_config, "model_type") and model_config.model_type == "btlm":
        if cfg.flash_attention:
            from axolotl.monkeypatch.btlm_attn_hijack_flash import (
                replace_btlm_attn_with_flash_attn,
            )

            replace_btlm_attn_with_flash_attn(cfg.base_model)

    if (
        hasattr(model_config, "model_type")
        and model_config.model_type == "stablelm_epoch"
    ):
        if cfg.flash_attention and cfg.sample_packing:
            from axolotl.monkeypatch.stablelm_attn_hijack_flash import (
                replace_stablelm_attn_with_flash_attn,
            )

            replace_stablelm_attn_with_flash_attn(cfg.base_model)

    if cfg.is_llama_derived_model and cfg.flash_attention and cfg.sample_packing:
        if cfg.device not in ["mps", "cpu"] and not inference:
            from axolotl.monkeypatch.llama_attn_hijack_flash import (
                replace_llama_attn_with_flash_attn,
            )

            LOG.info("patching with flash attention for sample packing")
            replace_llama_attn_with_flash_attn(
                packed=cfg.sample_packing,
                cross_entropy=cfg.flash_attn_cross_entropy,
                rms_norm=cfg.flash_attn_rms_norm,
            )
    elif cfg.is_llama_derived_model and cfg.xformers_attention:
        from axolotl.monkeypatch.llama_attn_hijack_xformers import (
            hijack_llama_attention,
        )

        LOG.info("patching with xformers attention")
        hijack_llama_attention()
    elif cfg.is_llama_derived_model and cfg.sdp_attention:
        from axolotl.monkeypatch.llama_attn_hijack_sdp import hijack_llama_sdp_attention

        LOG.info("patching with sdp attention")
        hijack_llama_sdp_attention()
    elif cfg.is_llama_derived_model and cfg.landmark_attention:
        from axolotl.monkeypatch.llama_landmark_attn import (
            MEM_TOKEN,
            patch_llama_with_landmark_attn,
        )

        LOG.info("patching with landmark attention")
        patch_llama_with_landmark_attn()

        # Note: This might overwrite previous additional_special_tokens
        tokenizer.add_special_tokens({"additional_special_tokens": [MEM_TOKEN]})

    if cfg.is_mistral_derived_model and cfg.flash_attention and cfg.sample_packing:
        from axolotl.monkeypatch.mistral_attn_hijack_flash import (
            replace_mistral_attn_with_flash_attn,
        )

        LOG.info("patching with flash attention")
        replace_mistral_attn_with_flash_attn(packed=cfg.sample_packing)

    if cfg.is_llama_derived_model and cfg.xpos_rope:
        from axolotl.monkeypatch.xpos_rope_llama_monkey_patch import (
            replace_llama_rope_with_xpos_rope,
        )

        LOG.info("patching with xpos rope")
        replace_llama_rope_with_xpos_rope()

    if (
        cfg.is_llama_derived_model
        and (cfg.max_packed_sequence_len or cfg.sample_packing)
        and not inference
    ):
        from axolotl.monkeypatch.llama_expand_mask import hijack_expand_mask

        LOG.info("patching _expand_mask")
        hijack_expand_mask()

    model_kwargs = {}

    model_kwargs["device_map"] = cfg.device_map
    model_kwargs["torch_dtype"] = cfg.torch_dtype

    if cfg.model_revision:
        model_kwargs["revision"] = cfg.model_revision
    if cfg.gptq:
        if not hasattr(model_config, "quantization_config"):
            LOG.warning("model config does not contain quantization_config information")
        else:
            if cfg.gptq_disable_exllama is not None:
                model_config.quantization_config[
                    "disable_exllama"
                ] = cfg.gptq_disable_exllama
            model_kwargs["quantization_config"] = GPTQConfig(
                **model_config.quantization_config
            )
    if cfg.adapter == "qlora" and cfg.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=cfg.torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    # sample packing uses custom FA2 patch
    if cfg.flash_attention and not cfg.sample_packing:
        if (
            cfg.is_llama_derived_model
            or cfg.is_falcon_derived_model
            or cfg.is_mistral_derived_model
        ):
            model_kwargs["use_flash_attention_2"] = True

    try:
        if cfg.is_llama_derived_model and not cfg.trust_remote_code and not cfg.gptq:
            from transformers import LlamaForCausalLM

            config_kwargs = {}
            if cfg.rope_scaling:
                config_kwargs["rope_scaling"] = cfg.rope_scaling
            config = LlamaConfig.from_pretrained(
                base_model_config,
                **config_kwargs,
            )
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                config=config,
                load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                **model_kwargs,
            )

            if cfg.flash_attention and not inference:
                from axolotl.monkeypatch.llama_attn_hijack_flash import (
                    replace_llama_mlp_with_swiglu,
                    replace_llama_qkv_with_fused,
                )

                if cfg.flash_attn_fuse_mlp:
                    LOG.info("patching with SwiGLU")
                    replace_llama_mlp_with_swiglu(model)

                if cfg.flash_attn_fuse_qkv:
                    LOG.info("patching with fused QKV")
                    replace_llama_qkv_with_fused(model)
        # elif model_type == "GPTNeoXForCausalLM" and cfg.flash_attention:
        #     This is a WIP, still an issue with the backward pass
        #     RuntimeError: grad can be implicitly created only for scalar outputs
        #     TODO: try config.sequence_parallel = False
        #     # https://github.com/HazyResearch/flash-attention/blob/40a25c8ee7465cf547b929cfa2937034e37bfce9/tests/models/test_gpt_neox.py#L12
        #     # https://github.com/HazyResearch/flash-attention/tree/main/training#model-components
        #     # add `**kwargs` to https://github.com/HazyResearch/flash-attention/blob/40a25c8ee7465cf547b929cfa2937034e37bfce9/flash_attn/models/gpt.py#L442
        #     from flash_attn.utils.pretrained import state_dict_from_pretrained
        #     from flash_attn.models.gpt import GPTLMHeadModel
        #     from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox, gpt_neox_config_to_gpt2_config
        #     from transformers import GPTNeoXConfig
        #     config = gpt_neox_config_to_gpt2_config(GPTNeoXConfig.from_pretrained(base_model))
        #     config.use_flash_attn = True
        #     config.fused_bias_fc = True
        #     config.fused_mlp = True  # GPT-NeoX-20B uses "gelu_fast"
        #     config.activation_function = "gelu_fast"
        #     config.fused_dropout_add_ln = True
        #     # config.residual_in_fp32 = True
        #
        #     model: GPTLMHeadModel = GPTLMHeadModel.from_pretrained(
        #         base_model,
        #         config,
        #         dtype=torch_dtype,
        #         device=cfg.device,
        #     )
        #     model.train() # sets to train instead of eval mode
        elif model_type == "MixFormerSequentialForCausalLM":
            from axolotl.models.phi import MixFormerSequentialForCausalLM

            model = MixFormerSequentialForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                **model_kwargs,
            )
        elif model_type and not cfg.trust_remote_code:
            if cfg.gptq:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    trust_remote_code=cfg.trust_remote_code or False,
                    **model_kwargs,
                )
            else:
                model = getattr(transformers, model_type).from_pretrained(
                    base_model,
                    load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                    load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                    trust_remote_code=cfg.trust_remote_code or False,
                    **model_kwargs,
                )
        else:
            config = AutoConfig.from_pretrained(
                base_model,
                trust_remote_code=cfg.trust_remote_code or False,
            )
            # Shouldn't be a problem most of the time. will obviously error if the model doesn't support this
            # when training starts
            if (
                hasattr(config, "max_seq_len")
                and config.max_seq_len
                and cfg.sequence_len > config.max_seq_len
            ):
                config.max_seq_len = cfg.sequence_len
                LOG.warning(f"increasing context length to {cfg.sequence_len}")
            elif (
                hasattr(config, "max_sequence_length")
                and config.max_sequence_length
                and cfg.sequence_len > config.max_sequence_length
            ):
                config.max_sequence_length = cfg.sequence_len
                LOG.warning(f"increasing context length to {cfg.sequence_len}")
            if cfg.gptq:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    config=config,
                    trust_remote_code=cfg.trust_remote_code or False,
                    **model_kwargs,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    config=config,
                    load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                    load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                    trust_remote_code=cfg.trust_remote_code or False,
                    **model_kwargs,
                )
    except Exception as err:  # pylint: disable=broad-exception-caught
        LOG.error(
            "Exception raised attempting to load model, retrying with AutoModelForCausalLM"
        )
        LOG.exception(err)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
            load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
            trust_remote_code=cfg.trust_remote_code or False,
            **model_kwargs,
        )

    # Dequantize for qlora
    # Adapt from https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
    if cfg.merge_lora and cfg.adapter == "qlora" and cfg.load_in_4bit:
        import copy
        import bitsandbytes as bnb
        from bitsandbytes.functional import dequantize_4bit
        from peft.utils import _get_submodules
        import gc

        LOG.info("dequantizing qlora model")
        dtype = torch.float16
        device = model.device
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, bnb.nn.Linear4bit):
                    # LOG.info(f"Dequantizing `{name}`...")
                    quant_state = copy.deepcopy(module.weight.quant_state)

                    quant_state[2] = dtype

                    weights = dequantize_4bit(
                        module.weight.data, quant_state=quant_state, quant_type="nf4"
                    ).to(dtype)

                    new_module = torch.nn.Linear(
                        module.in_features, module.out_features, bias=None, dtype=dtype
                    )
                    new_module.weight = torch.nn.Parameter(weights)
                    new_module.to(device=device, dtype=dtype)

                    parent, target, target_name = _get_submodules(model, name)
                    setattr(parent, target_name, new_module)
                    del module

            gc.collect()
            torch.cuda.empty_cache()

            model.is_loaded_in_4bit = False
            model.quantization_method = None
            model.is_quantized = False
            delattr(model.config, "quantization_config")
            # If your model is too large, you may need to call `model.cpu()` to merge on CPU
            model.to("cpu")

    embeddings_len = (
        math.ceil(len(tokenizer) / 32) * 32
        if cfg.resize_token_embeddings_to_32x
        else len(tokenizer)
    )
    if model.get_input_embeddings().num_embeddings < embeddings_len:
        model.resize_token_embeddings(embeddings_len)
    else:
        model.tie_weights()

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings
        and cfg.sequence_len > model.config.max_position_embeddings
    ):
        LOG.warning(
            f"increasing model.config.max_position_embeddings from {model.config.max_position_embeddings} to {cfg.sequence_len}"
        )
        model.config.max_position_embeddings = cfg.sequence_len

    if (
        hasattr(model.config, "bos_token_id")
        and model.config.bos_token_id
        and model.config.bos_token_id != tokenizer.bos_token_id
    ):
        model.config.bos_token_id = tokenizer.bos_token_id

    if (
        hasattr(model.config, "eos_token_id")
        and model.config.eos_token_id
        and model.config.eos_token_id != tokenizer.eos_token_id
    ):
        model.config.eos_token_id = tokenizer.eos_token_id

    if model.device.type == "cuda":
        log_gpu_memory_usage(LOG, "after model load", model.device)

    # make sure these are fp32 per Ramesh et al. (2021)
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(torch.float32)
        if model_config.model_type == "btlm":
            # don't upcast lm_head for btlm
            continue
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(torch.float32)

    needs_fa2_dtype = cfg.adapter or cfg.fsdp
    if (cfg.adapter == "lora" and load_in_8bit) or (
        cfg.adapter == "qlora" and cfg.load_in_4bit
    ):
        LOG.info("converting PEFT model w/ prepare_model_for_kbit_training")
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
        needs_fa2_dtype = True

    # LlamaRMSNorm layers are in fp32 after kbit_training or full finetune, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    if needs_fa2_dtype or (cfg.flash_attention and cfg.is_llama_derived_model):
        LOG.info("converting modules to %s for flash attention", cfg.torch_dtype)
        for name, module in model.named_modules():
            if "norm" in name:
                module.to(cfg.torch_dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module.to(cfg.torch_dtype)

    if cfg.logging_topk is not None:
        import wandb

        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs["labels"]
            loss = outputs.loss

            logs = {}

            # shift
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            not_ignore_mask = labels != -100

            for k in range(1, cfg.logging_topk + 1):
                # compute top-k accuracy
                _, topk_indices = logits.topk(k, dim=-1)
                correct = topk_indices == labels.unsqueeze(-1)
                correct = correct.sum(-1)
                correct = correct.masked_select(not_ignore_mask).sum()
                accuracy = correct.float() / not_ignore_mask.sum()
                logs[f"top_{k}"] = accuracy

            if model.training:
                prefix = "train"
            else:
                prefix = "eval"

            logs = {f"{prefix}/{k}": v for k, v in logs.items()}
            if self.state.is_world_process_zero:
                wandb.log(
                    {
                        **logs,
                        "train/global_step": self.state.global_step,
                    }
                )

            return (loss, outputs) if return_outputs else loss
        transformers.trainer.Trainer.compute_loss = compute_loss

    # Add support for Hydra (https://github.com/FasterDecoding/Hydra)
    if cfg.hydra_num_heads is not None:
        from transformers import LlamaForCausalLM, MistralForCausalLM

        assert isinstance(
            model, (LlamaForCausalLM, MistralForCausalLM)
        ), "Hydra is only supported for Llama and Mistral models for now"

        LOG.info(
            f"using Hydra with {cfg.hydra_num_heads} heads, {cfg.hydra_num_layers} layers, {cfg.hydra_decay_coefficient} decay coefficient, {cfg.hydra_heads_coefficient} heads coefficient, {cfg.hydra_scheduler} scheduler, {cfg.hydra_logging} logging"
        )

        add_hydra_heads(
            model,
            hydra_num_heads=cfg.hydra_num_heads,
            hydra_num_layers=cfg.hydra_num_layers,
            grounded_heads=cfg.grounded_heads,
            hydra_head_arch=cfg.hydra_head_arch,
        )

        if cfg.flash_attention and cfg.hydra_heads and not inference:
            if cfg.flash_attn_fuse_mlp:
                LOG.info("patching with SwiGLU")
                replace_llama_mlp_with_swiglu(model.hydra_head.prefix_embeding_layer)

            if cfg.flash_attn_fuse_qkv:
                LOG.info("patching with fused QKV")
                replace_llama_qkv_with_fused(model.hydra_head.prefix_embeding_layer)


        replace_compute_loss(
            hydra_heads_coefficient=cfg.hydra_heads_coefficient,
            hydra_decay_coefficient=cfg.hydra_decay_coefficient,
            hydra_scheduler=cfg.hydra_scheduler,
            hydra_logging=cfg.hydra_logging,
            hydra_only_heads=cfg.hydra_only_heads,
            hydra_distillation_regularization=cfg.hydra_distillation_regularization,
            hydra_self_distillation=cfg.hydra_self_distillation,
        )

        if cfg.hydra_lr_multiplier != 1:
            LOG.info(f"Using Hydra LR multiplier {cfg.hydra_lr_multiplier}")
            replace_create_optimizer(
                hydra_lr_multiplier=cfg.hydra_lr_multiplier,
            )

        # if cfg.adapter in ["lora", "qlora"]:
        #     # Add hydra heads to cfg.lora_modules_to_save
        #     if cfg.lora_modules_to_save is None:
        #         cfg.lora_modules_to_save = []
        #     cfg.lora_modules_to_save.append("hydra_head")
        #     # for i in range(cfg.hydra_num_heads):
        #     #     cfg.lora_modules_to_save.append(f"hydra_head.{i}")
        #     # for name, module in model.hydra_head.named_modules():
        #     #     if isinstance(module, torch.nn.Linear):
        #     #         cfg.lora_modules_to_save.append(f"hydra_head.{name}")
        #     # cfg.lora_modules_to_save.append("lm_head")

    model, lora_config = load_adapter(model, cfg, cfg.adapter)

    if cfg.hydra_num_heads is not None and (
        cfg.hydra_only_heads or cfg.hydra_num_unfreeze_layers > 0
    ):
        LOG.info("Freeze layers!")
        for param in model.parameters():
            param.requires_grad = False
        # Leave the last hydra_num_unfreeze_layers layers trainable
        if cfg.hydra_num_unfreeze_layers > 0:
            for layer in model.model.layers[-cfg.hydra_num_unfreeze_layers :]:
                LOG.info(f"Unfreezing layer {layer}")
                for param in layer.parameters():
                    param.requires_grad = True
            # Leave the last hydra_num_unfreeze_layers layers trainable to ensure the gradient can pass through
            for param in model.model.norm.parameters():
                param.requires_grad = True

        for param in model.hydra_head.parameters():
            param.requires_grad = True

        if not cfg.hydra_only_heads:
            for param in model.lm_head.parameters():
                param.requires_grad = True

        if cfg.gradient_checkpointing:
            # https://github.com/huggingface/transformers/issues/21381#issuecomment-1666498410
            from functools import partial

            notfailing_checkpoint = partial(
                torch.utils.checkpoint.checkpoint, use_reentrant=False
            )
            torch.utils.checkpoint.checkpoint = notfailing_checkpoint

    if cfg.ddp and not load_in_8bit:
        model.to(f"cuda:{cfg.local_rank}")

    if (
        torch.cuda.device_count() > 1
        and int(os.getenv("WORLD_SIZE", "1")) > 1
        and (cfg.load_in_4bit)
    ):
        # llama is PROBABLY model parallelizable, but the default isn't that it is
        # so let's only set it for the 4bit, see
        # https://github.com/johnsmith0031/alpaca_lora_4bit/blob/08b3fca4a4a9e0d3945be1bab4529f100a428636/finetune.py#L130-L133
        setattr(model, "is_parallelizable", True)
        setattr(model, "model_parallel", True)

    requires_grad = []
    for name, param in model.named_parameters(recurse=True):
        if param.requires_grad:
            requires_grad.append(f"{name}: {param.requires_grad}")
    if len(requires_grad) == 0:
        LOG.warning("there are no parameters that require gradient updates")
    model.config.use_cache = False

    if cfg.flash_optimum:
        model = BetterTransformer.transform(model)

    if cfg.adapter is not None:
        log_gpu_memory_usage(LOG, "after adapters", model.device)

    # TODO resume_from_checkpoint handling
    return model, lora_config


def load_adapter(model, cfg, adapter, inference=False):
    # type: (PreTrainedModel, DictDefault, Optional[str], bool) -> Tuple[PreTrainedModel, Optional[PeftConfig]]

    if adapter is None:
        return model, None
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if adapter in ["lora", "qlora"]:
        return load_lora(model, cfg, inference=inference)
    if adapter == "llama-adapter":
        return load_llama_adapter(model, cfg)

    raise NotImplementedError(f"{adapter} peft adapter not available")


def load_llama_adapter(model, cfg):
    # type: (PreTrainedModel, DictDefault) -> Tuple[PreTrainedModel, Optional[PeftConfig]]
    from peft import AdaptionPromptConfig, PeftModel, get_peft_model

    peft_config = AdaptionPromptConfig(
        adapter_layers=cfg.peft_adapter.layers,  # layers (L)
        adapter_len=cfg.peft_adapter.len,  # prompt length (K)
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        LOG.debug("Loading pretained PEFT - llama_adapter")
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            torch_dtype=torch.float16,
        )
    else:
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model, peft_config


def find_all_linear_names(model):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def load_lora(model, cfg, inference=False):
    # type: (PreTrainedModel, DictDefault, bool) -> Tuple[PreTrainedModel, Optional[PeftConfig]]

    from peft import LoraConfig, PeftModel, get_peft_model

    lora_target_modules = list(cfg.lora_target_modules or [])

    if cfg.lora_target_linear:
        linear_names = find_all_linear_names(model)
        LOG.info(f"found linear modules: {repr(linear_names)}")
        lora_target_modules = list(set(lora_target_modules + linear_names))

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out,
        modules_to_save=cfg.lora_modules_to_save if cfg.lora_modules_to_save else None,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        LOG.debug("Loading pretained PEFT - LoRA")
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            is_trainable=(not inference),
        )
    else:
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, lora_config
