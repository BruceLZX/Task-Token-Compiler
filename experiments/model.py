"""Models and baselines for Task/Sensor Token Compiler experiments."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .compiler import MLPTokenCompiler, SensorMetaEncoder, SensorTokenCompiler
else:  # Support running scripts directly from experiments/.
    from compiler import MLPTokenCompiler, SensorMetaEncoder, SensorTokenCompiler


class PatchTransformerBackbone(nn.Module):
    """Small prefix-capable TSFM-style encoder for controlled experiments."""

    def __init__(self, config):
        super().__init__()
        cfg = config.backbone
        self.d_model = cfg.d_model
        self.patch_len = cfg.patch_len
        self.patch_embed = nn.Conv1d(1, cfg.d_model, kernel_size=cfg.patch_len, stride=cfg.patch_len)
        max_patches = max(8, config.data.window_len // cfg.patch_len + 32)
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, prefix_tokens: Optional[torch.Tensor] = None):
        patches = self.patch_embed(x).transpose(1, 2)
        if prefix_tokens is not None:
            tokens = torch.cat([prefix_tokens, patches], dim=1)
        else:
            tokens = patches
        pos = self.pos_embed[:, : tokens.shape[1], :]
        tokens = tokens + pos
        encoded = self.norm(self.encoder(tokens))
        return encoded


class MomentPrefixBackbone(nn.Module):
    """Frozen official MOMENT encoder with per-sample prefix token injection."""

    MODEL_IDS = {
        "small": "AutonLab/MOMENT-1-small",
        "base": "AutonLab/MOMENT-1-base",
        "large": "AutonLab/MOMENT-1-large",
    }

    def __init__(self, config):
        super().__init__()
        try:
            from momentfm import MOMENTPipeline
            from momentfm.utils.masking import Masking
        except ImportError as exc:
            raise ImportError(
                "momentfm is required for backbone=moment_prefix. Install "
                "experiments/requirements.txt in Python 3.11, or run inside the "
                "Space MOMENT venv."
            ) from exc

        variant = config.backbone.moment_variant
        model_id = self.MODEL_IDS.get(variant, variant)
        self.moment = MOMENTPipeline.from_pretrained(
            model_id,
            model_kwargs={"task_name": "embedding"},
        )
        self.moment.init()
        self.masking = Masking
        self.d_model = int(self.moment.config.d_model)
        self.patch_len = int(self.moment.patch_len)

        for param in self.moment.parameters():
            param.requires_grad = False
        self.moment.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.moment.eval()
        return self

    def forward(self, x: torch.Tensor, prefix_tokens: Optional[torch.Tensor] = None):
        batch_size, n_channels, seq_len = x.shape
        input_mask = torch.ones((batch_size, seq_len), device=x.device, dtype=x.dtype)

        x_enc = self.moment.normalizer(x=x, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        x_enc = self.moment.tokenizer(x=x_enc)
        enc_in = self.moment.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(batch_size * n_channels, n_patches, self.d_model)
        attention_mask = self.masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        ).repeat_interleave(n_channels, dim=0)

        n_prefix = 0
        if prefix_tokens is not None:
            n_prefix = prefix_tokens.shape[1]
            prefix = prefix_tokens.repeat_interleave(n_channels, dim=0)
            enc_in = torch.cat([prefix, enc_in], dim=1)
            prefix_mask = torch.ones(
                enc_in.shape[0],
                n_prefix,
                device=x.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.moment.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        if n_prefix:
            enc_out = enc_out[:, n_prefix:, :]
        enc_out = enc_out.reshape(batch_size, n_channels, n_patches, self.d_model)

        # Return patch tokens after channel averaging. The prefix has influenced
        # the frozen encoder through attention, but is not used as a readout shortcut.
        return enc_out.mean(dim=1)


class AdapterBlock(nn.Module):
    """Tiny bottleneck adapter used as a PEFT baseline."""

    def __init__(self, d_model: int, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(x)))


class MetadataMLP(nn.Module):
    """Post-backbone metadata side-feature baseline."""

    def __init__(self, config):
        super().__init__()
        self.encoder = SensorMetaEncoder(config)
        self.proj = nn.Linear(config.compiler.hidden_dim, config.backbone.d_model)

    def forward(self, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.proj(self.encoder(metadata))


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for linear layers."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * (self.alpha / self.rank)

    @property
    def weight(self):
        # PyTorch TransformerEncoderLayer fast paths read .weight/.bias directly.
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def apply_lora_to_transformer_ffn(module: nn.Module, rank: int = 8) -> None:
    """Apply LoRA to Transformer feed-forward Linear layers."""
    for child_name, child in module.named_children():
        if isinstance(child, nn.TransformerEncoderLayer):
            child.linear1 = LoRALinear(child.linear1, rank=rank)
            child.linear2 = LoRALinear(child.linear2, rank=rank)
        else:
            apply_lora_to_transformer_ffn(child, rank=rank)


class TaskSensorModel(nn.Module):
    """
    Unified model for proposed method and reviewer baselines.

    Methods:
        compiler_prefix: metadata -> compiled prefix tokens -> frozen encoder.
        compiler_concat: post-backbone concatenation control.
        frozen_probe: no metadata, no tokens.
        metadata_mlp: post-backbone metadata side features.
        random_prefix: fixed random prefix control.
        per_sensor_prompt: learned prompt table indexed by sensor_type.
        per_sensor_task_prompt: stronger prompt lookup indexed by sensor-task pair.
        mlp_token_compiler: simple MLP metadata-to-prefix baseline.
        sensor_id_prompt: alias for per_sensor_prompt.
        adapter: frozen backbone plus trainable bottleneck adapter after pooling.
        lora: LoRA on transformer feed-forward layers.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.method = config.method
        if config.backbone.backbone == "patch_transformer":
            self.backbone = PatchTransformerBackbone(config)
        elif config.backbone.backbone == "moment_prefix":
            self.backbone = MomentPrefixBackbone(config)
            config.backbone.d_model = self.backbone.d_model
        else:
            raise NotImplementedError(f"Unknown backbone: {config.backbone.backbone}")
        d_model = config.backbone.d_model
        num_classes = config.data.num_classes

        self.compiler: Optional[SensorTokenCompiler] = None
        self.mlp_token_compiler: Optional[MLPTokenCompiler] = None
        self.metadata_mlp: Optional[MetadataMLP] = None
        self.adapter: Optional[AdapterBlock] = None

        num_prefix_tokens = config.compiler.num_sensor_tokens + config.compiler.num_task_tokens
        if self.method == "compiler_prefix" or self.method == "compiler_concat":
            self.compiler = SensorTokenCompiler(config)
        elif self.method == "mlp_token_compiler":
            self.mlp_token_compiler = MLPTokenCompiler(config)
        elif self.method == "metadata_mlp":
            self.metadata_mlp = MetadataMLP(config)
        elif self.method == "random_prefix":
            self.random_prefix = nn.Parameter(
                torch.randn(1, num_prefix_tokens, d_model) * 0.02,
                requires_grad=False,
            )
        elif self.method in {"per_sensor_prompt", "sensor_id_prompt"}:
            self.prompt_table = nn.Embedding(32, num_prefix_tokens * d_model)
        elif self.method == "per_sensor_task_prompt":
            self.prompt_table = nn.Embedding(32 * 16, num_prefix_tokens * d_model)
        elif self.method == "adapter":
            self.adapter = AdapterBlock(d_model)
        elif self.method == "lora":
            if config.backbone.backbone != "patch_transformer":
                raise NotImplementedError("lora is currently wired only for patch_transformer")
            apply_lora_to_transformer_ffn(self.backbone.encoder, rank=config.backbone.lora_rank)
        elif self.method != "frozen_probe":
            raise ValueError(f"Unknown method: {self.method}")

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )

        self._set_trainable_parameters()

    def _set_trainable_parameters(self) -> None:
        if self.config.backbone.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.method == "lora":
            # LoRA params remain trainable; frozen base params remain frozen.
            for name, p in self.backbone.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

    def _prefix_tokens(self, x: torch.Tensor, metadata: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        bsz = x.shape[0]
        if self.method == "compiler_prefix":
            assert self.compiler is not None
            return self.compiler(metadata, x)
        if self.method == "mlp_token_compiler":
            assert self.mlp_token_compiler is not None
            return self.mlp_token_compiler(metadata, x)
        if self.method == "random_prefix":
            return self.random_prefix.expand(bsz, -1, -1)
        if self.method in {"per_sensor_prompt", "sensor_id_prompt"}:
            sensor_ids = metadata["sensor_type"].long().view(bsz).to(x.device)
            raw = self.prompt_table(sensor_ids)
            n = self.config.compiler.num_sensor_tokens + self.config.compiler.num_task_tokens
            return raw.view(bsz, n, self.config.backbone.d_model)
        if self.method == "per_sensor_task_prompt":
            sensor_ids = metadata["sensor_type"].long().view(bsz).to(x.device)
            task_ids = metadata["task_description"].long().view(bsz).to(x.device)
            pair_ids = sensor_ids * 16 + task_ids
            raw = self.prompt_table(pair_ids)
            n = self.config.compiler.num_sensor_tokens + self.config.compiler.num_task_tokens
            return raw.view(bsz, n, self.config.backbone.d_model)
        return None

    def forward(
        self,
        x: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
        return_tokens: bool = False,
    ):
        prefix = self._prefix_tokens(x, metadata)
        encoded = self.backbone(x, prefix_tokens=prefix)
        pooled = encoded.mean(dim=1)

        compiled_tokens = None
        if self.method == "compiler_concat":
            assert self.compiler is not None
            compiled_tokens = self.compiler(metadata, x)
            pooled = pooled + compiled_tokens.mean(dim=1)
        elif self.method == "metadata_mlp":
            assert self.metadata_mlp is not None
            pooled = pooled + self.metadata_mlp(metadata)
        elif self.method == "adapter":
            assert self.adapter is not None
            pooled = self.adapter(pooled)
        elif self.method == "compiler_prefix":
            compiled_tokens = prefix

        logits = self.head(pooled)
        if return_tokens:
            return logits, compiled_tokens
        return logits

    def regularization_loss(self, metadata: Dict[str, torch.Tensor], tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if self.compiler is None or tokens is None:
            return next(self.parameters()).new_tensor(0.0)
        weight = float(self.config.compiler.structure_loss_weight)
        if weight <= 0:
            return tokens.new_tensor(0.0)
        return weight * self.compiler.token_structure_loss(metadata, tokens)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
