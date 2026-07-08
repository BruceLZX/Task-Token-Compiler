"""Task/Sensor Token Compiler modules."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """Continuous scalar encoding for rates and durations."""

    def __init__(self, num_bands: int = 8, max_freq: float = 1000.0):
        super().__init__()
        freqs = torch.logspace(0, torch.log10(torch.tensor(max_freq)), num_bands)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().view(-1, 1)
        # Log-scale inputs make 4 Hz and 700 Hz separable without huge values.
        z = torch.log1p(torch.clamp(x, min=0.0)) / torch.log(torch.tensor(1001.0, device=x.device))
        angles = 2 * torch.pi * z * self.freqs.view(1, -1).to(x.device)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class SignalSummaryEncoder(nn.Module):
    """Tiny waveform-statistics encoder used only by the compiler."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        mono = x.mean(dim=1)
        mean = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)
        amin = mono.amin(dim=-1, keepdim=True)
        amax = mono.amax(dim=-1, keepdim=True)
        energy = (mono**2).mean(dim=-1, keepdim=True)
        diff = mono[:, 1:] - mono[:, :-1]
        diff_std = diff.std(dim=-1, keepdim=True)
        zcr = ((mono[:, 1:] * mono[:, :-1]) < 0).float().mean(dim=-1, keepdim=True)
        autocorr = (mono[:, 1:] * mono[:, :-1]).mean(dim=-1, keepdim=True)
        fft = torch.fft.rfft(mono.float(), dim=-1).abs()
        n = fft.shape[-1]
        bands = []
        for start, end in [(1, n // 16), (n // 16, n // 8), (n // 8, n // 4), (n // 4, n // 2)]:
            start = max(1, start)
            end = max(start + 1, end)
            bands.append(fft[:, start:end].mean(dim=-1, keepdim=True))
        stats = torch.cat([mean, std, amin, amax, energy, diff_std, zcr, autocorr] + bands, dim=-1)
        return self.net(stats)


class SensorMetaEncoder(nn.Module):
    """Encode structured sensor/task metadata into one context vector."""

    CARDINALITIES = {
        "sensor_type": 32,
        "measured_process": 16,
        "body_location": 16,
        "physical_unit": 16,
        "channel_layout": 8,
        "task_type": 8,
        "task_description": 16,
    }

    NUMERIC_FIELDS = ("sampling_rate", "window_duration")

    def __init__(self, config):
        super().__init__()
        self.cfg = config.sensor_meta
        self.field_dropout = float(self.cfg.field_dropout)
        cat_dim = int(self.cfg.categorical_dim)
        num_dim = int(self.cfg.numeric_dim)

        self.embeddings = nn.ModuleDict()
        for field, cardinality in self.CARDINALITIES.items():
            if self._field_enabled(field):
                self.embeddings[field] = nn.Embedding(cardinality, cat_dim)

        self.numeric = nn.ModuleDict()
        for field in self.NUMERIC_FIELDS:
            if self._field_enabled(field):
                self.numeric[field] = nn.Sequential(
                    FourierFeatures(num_bands=max(1, num_dim // 2)),
                    nn.Linear(num_dim, cat_dim),
                    nn.LayerNorm(cat_dim),
                    nn.GELU(),
                )

        total = cat_dim * (len(self.embeddings) + len(self.numeric))
        self.out_dim = max(cat_dim, total)
        self.proj = nn.Sequential(
            nn.Linear(self.out_dim, config.compiler.hidden_dim),
            nn.LayerNorm(config.compiler.hidden_dim),
            nn.GELU(),
        )

    def _field_enabled(self, field: str) -> bool:
        if field == "sensor_type" and self.cfg.drop_sensor_name:
            return False
        if field == "task_description" and self.cfg.drop_task_description:
            return False
        return bool(getattr(self.cfg, field, False))

    def _maybe_drop(self, emb: torch.Tensor) -> torch.Tensor:
        if not self.training or self.field_dropout <= 0:
            return emb
        keep = torch.rand(emb.shape[0], 1, device=emb.device) > self.field_dropout
        return emb * keep.float()

    def forward(self, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        pieces = []
        device = next(self.parameters()).device
        batch = next(iter(metadata.values())).shape[0]
        for field, emb in self.embeddings.items():
            values = metadata[field].to(device).long().view(batch)
            pieces.append(self._maybe_drop(emb(values)))
        for field, enc in self.numeric.items():
            values = metadata[field].to(device).float().view(batch)
            pieces.append(self._maybe_drop(enc(values)))
        if not pieces:
            pieces.append(torch.zeros(batch, self.out_dim, device=device))
        x = torch.cat(pieces, dim=-1)
        if x.shape[-1] < self.out_dim:
            x = F.pad(x, (0, self.out_dim - x.shape[-1]))
        return self.proj(x)


class SensorTokenCompiler(nn.Module):
    """
    Compile metadata and optional signal summaries into prefix adaptation tokens.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = config.compiler
        self.meta_encoder = SensorMetaEncoder(config)
        self.use_signal_summary = bool(cfg.use_signal_summary)
        if self.use_signal_summary:
            self.signal_encoder = SignalSummaryEncoder(cfg.hidden_dim)
        else:
            self.signal_encoder = None

        self.num_tokens = int(cfg.num_sensor_tokens + cfg.num_task_tokens)
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_tokens, cfg.hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.to_backbone_dim = nn.Linear(cfg.hidden_dim, config.backbone.d_model)

    def forward(
        self,
        metadata: Dict[str, torch.Tensor],
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = next(iter(metadata.values())).shape[0]
        meta = self.meta_encoder(metadata).unsqueeze(1)
        contexts = [meta]
        if self.signal_encoder is not None and x is not None:
            contexts.append(self.signal_encoder(x).unsqueeze(1))
        queries = self.query_tokens.expand(batch, -1, -1)
        tokens = torch.cat(contexts + [queries], dim=1)
        compiled = self.transformer(tokens)
        prefix = compiled[:, -self.num_tokens :, :]
        return self.to_backbone_dim(prefix)

    def token_structure_loss(self, metadata: Dict[str, torch.Tensor], tokens: torch.Tensor) -> torch.Tensor:
        """Encourage token distances to agree with simple metadata distances."""
        if tokens.shape[0] < 3:
            return tokens.new_tensor(0.0)
        pooled = F.normalize(tokens.mean(dim=1), dim=-1)
        token_dist = torch.cdist(pooled, pooled, p=2)
        meta_parts = []
        for key in ("sensor_type", "measured_process", "body_location", "physical_unit", "task_description"):
            if key in metadata:
                value = metadata[key].float().view(-1, 1).to(tokens.device)
                meta_parts.append(value / (value.max().clamp(min=1.0)))
        for key in ("sampling_rate", "window_duration"):
            if key in metadata:
                value = torch.log1p(metadata[key].float().view(-1, 1).to(tokens.device))
                meta_parts.append(value / (value.max().clamp(min=1.0)))
        if not meta_parts:
            return tokens.new_tensor(0.0)
        meta = torch.cat(meta_parts, dim=-1)
        meta_dist = torch.cdist(meta, meta, p=1)
        token_dist = token_dist / token_dist.detach().mean().clamp(min=1e-6)
        meta_dist = meta_dist / meta_dist.detach().mean().clamp(min=1e-6)
        mask = ~torch.eye(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        return F.smooth_l1_loss(token_dist[mask], meta_dist[mask])


class MLPTokenCompiler(nn.Module):
    """Simple metadata-to-prefix baseline for over-engineering checks."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.meta_encoder = SensorMetaEncoder(config)
        self.use_signal_summary = bool(config.compiler.use_signal_summary)
        if self.use_signal_summary:
            self.signal_encoder = SignalSummaryEncoder(config.compiler.hidden_dim)
            in_dim = config.compiler.hidden_dim * 2
        else:
            self.signal_encoder = None
            in_dim = config.compiler.hidden_dim
        self.num_tokens = int(config.compiler.num_sensor_tokens + config.compiler.num_task_tokens)
        out_dim = self.num_tokens * config.backbone.d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.compiler.hidden_dim),
            nn.LayerNorm(config.compiler.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.compiler.dropout),
            nn.Linear(config.compiler.hidden_dim, out_dim),
        )

    def forward(self, metadata: Dict[str, torch.Tensor], x: Optional[torch.Tensor] = None) -> torch.Tensor:
        meta = self.meta_encoder(metadata)
        pieces = [meta]
        if self.signal_encoder is not None and x is not None:
            pieces.append(self.signal_encoder(x))
        z = torch.cat(pieces, dim=-1)
        tokens = self.net(z)
        return tokens.view(z.shape[0], self.num_tokens, self.config.backbone.d_model)
