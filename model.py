import math
from typing import NamedTuple

import torch
from torch import nn

from action_encoding import (
    ACTION_VOCAB_SIZE,
    TOKEN_TO_ID,
    ActionHeadType,
    TOKENS_BY_HEAD,
)
from global_encoding import GLOBAL_FEATURE_NAMES
from kinds import AnimalType, CropType, TileKind
from observation_encoding import UNIT_FEATURE_NAMES
from tile_encoding import NUMERIC_FEATURE_NAMES


def embedding_dim(num_categories: int) -> int:
    """Kategori sayısına göre küçük bir embedding boyutu seçer."""
    return max(2, math.ceil(math.log2(num_categories)) + 1)


class TileEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        num_kinds = len(TileKind)
        num_crops = len(CropType) + 1
        num_animals = len(AnimalType) + 1

        self.kind_dim = embedding_dim(num_kinds)
        self.crop_dim = embedding_dim(num_crops)
        self.animal_dim = embedding_dim(num_animals)

        self.kind_embedding = nn.Embedding(
            num_embeddings=num_kinds,
            embedding_dim=self.kind_dim,
        )
        self.crop_embedding = nn.Embedding(
            num_embeddings=num_crops,
            embedding_dim=self.crop_dim,
            padding_idx=0,
        )
        self.animal_embedding = nn.Embedding(
            num_embeddings=num_animals,
            embedding_dim=self.animal_dim,
            padding_idx=0,
        )

    @property
    def output_dim(self) -> int:
        return self.kind_dim + self.crop_dim + self.animal_dim

    def forward(
        self,
        kind_ids: torch.Tensor,
        crop_ids: torch.Tensor,
        animal_ids: torch.Tensor,
    ) -> torch.Tensor:
        kind_vectors = self.kind_embedding(kind_ids)
        crop_vectors = self.crop_embedding(crop_ids)
        animal_vectors = self.animal_embedding(animal_ids)

        return torch.cat(
            [kind_vectors, crop_vectors, animal_vectors],
            dim=-1,
        )


class TileEncoder(nn.Module):
    """Embedding ve numeric bilgileri aynı tile üzerinde birleştirip işler."""

    def __init__(self, output_channels: int = 32) -> None:
        super().__init__()

        self.embedding = TileEmbedding()
        self.numeric_channels = len(NUMERIC_FEATURE_NAMES)
        self.input_channels = self.embedding.output_dim + self.numeric_channels
        self.output_channels = output_channels

        # Kernel 1 olduğu için her (y, x) tile'ini bağımsız işler.
        # Komşu tile bilgileri henüz birbirine karışmaz.
        self.tile_mixer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=output_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
        )

    def combine_features(
        self,
        categorical_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        """Aynı koordinattaki embedding ve numeric kanalları birleştirir."""
        kind_ids = categorical_ids[:, 0]
        crop_ids = categorical_ids[:, 1]
        animal_ids = categorical_ids[:, 2]

        embedded = self.embedding(kind_ids, crop_ids, animal_ids)
        embedded = embedded.permute(0, 3, 1, 2)

        return torch.cat([embedded, numeric_features], dim=1)

    def forward(
        self,
        categorical_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        combined_features = self.combine_features(
            categorical_ids,
            numeric_features,
        )
        return self.tile_mixer(combined_features)


class ResidualSpatialBlock(nn.Module):
    """Kanal sayısını koruyan iki 3x3 conv'lu residual blok."""

    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.block(features))


class SpatialEncoder(nn.Module):
    """Tile'ları 10x10 tarla ölçeğinde birlikte işler.

    Simetrik 3x3 conv yığını kullanılır. Beş conv katmanı 11x11 alıcı alan
    verir; bu 10x10 tarlayı kapsar ve tek katmanlı geniş kernel'in aksine
    komşuluk yapısını kademeli olarak kurar. Farm ölçeğindeki global özet
    ayrıca ``FarmPooling`` üzerinden taşınır.
    """

    def __init__(
        self,
        input_channels: int = 32,
        hidden_channels: int = 32,
        output_channels: int = 64,
        num_blocks: int = 2,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        if num_blocks < 1:
            raise ValueError("num_blocks en az 1 olmalı.")

        self.output_channels = output_channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *(
                ResidualSpatialBlock(hidden_channels, num_groups=num_groups)
                for _ in range(num_blocks)
            )
        )
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=output_channels),
            nn.ReLU(),
        )

    def forward(self, tile_features: torch.Tensor) -> torch.Tensor:
        return self.projection(self.blocks(self.stem(tile_features)))


class FarmEncoder(nn.Module):
    """Bir farmın tile ve worker konumlarını spatial haritaya çevirir."""

    def __init__(self) -> None:
        super().__init__()

        self.tile_encoder = TileEncoder(output_channels=32)
        self.worker_channels = 2

        self.farm_mixer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.tile_encoder.output_channels + self.worker_channels,
                out_channels=32,
                kernel_size=1,
            ),
            nn.ReLU(),
        )
        self.spatial_encoder = SpatialEncoder(input_channels=32)
        self.output_channels = self.spatial_encoder.output_channels

    def forward(
        self,
        categorical_ids: torch.Tensor,
        numeric_features: torch.Tensor,
        worker_positions: torch.Tensor,
    ) -> torch.Tensor:
        tile_features = self.tile_encoder(
            categorical_ids,
            numeric_features,
        )
        farm_features = torch.cat(
            [tile_features, worker_positions],
            dim=1,
        )
        farm_features = self.farm_mixer(farm_features)
        return self.spatial_encoder(farm_features)


class FarmPooling(nn.Module):
    """10x10 farm haritasını sabit boyutlu bir özet vektöre çevirir."""

    def __init__(
        self,
        input_channels: int = 64,
        output_dim: int = 64,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.projection = nn.Sequential(
            nn.Linear(input_channels * 2, output_dim),
            nn.ReLU(),
        )

    def forward(self, farm_map: torch.Tensor) -> torch.Tensor:
        average = farm_map.mean(dim=(2, 3))
        maximum = farm_map.amax(dim=(2, 3))
        return self.projection(torch.cat([average, maximum], dim=1))


class GlobalEncoder(nn.Module):
    """Market, ekonomi, zaman ve envanter vektörünü işler."""

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ) -> None:
        super().__init__()

        self.input_dim = len(GLOBAL_FEATURE_NAMES)
        self.output_dim = output_dim

        # Çıkış Tanh: işaretli özet, aralık [-1, 1].
        # ReLU/GELU gibi negatifleri sıfırlamaz; saturasyon Tanh'ın bedeli.
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        return self.network(global_features)


class StateEncoding(NamedTuple):
    own_map: torch.Tensor
    opponent_map: torch.Tensor
    own_summary: torch.Tensor
    opponent_summary: torch.Tensor
    global_summary: torch.Tensor
    policy_features: torch.Tensor
    state_vector: torch.Tensor


class StateEncoder(nn.Module):
    """Own farm, opponent farm ve global kolları ortak state temsilinde birleştirir."""

    def __init__(self) -> None:
        super().__init__()

        # Aynı FarmEncoder iki farm için de çağrılır; ağırlıklar ortaktır.
        self.farm_encoder = FarmEncoder()
        self.farm_pooling = FarmPooling(
            input_channels=self.farm_encoder.output_channels,
            output_dim=64,
        )
        self.global_encoder = GlobalEncoder(output_dim=64)

        self.policy_channels = 64 + 64 + 64
        self.state_dim = 64 + 64 + 64 + 64

    def forward(
        self,
        own_categorical_ids: torch.Tensor,
        own_numeric_features: torch.Tensor,
        own_worker_positions: torch.Tensor,
        opponent_categorical_ids: torch.Tensor,
        opponent_numeric_features: torch.Tensor,
        opponent_worker_positions: torch.Tensor,
        global_features: torch.Tensor,
    ) -> StateEncoding:
        own_map = self.farm_encoder(
            own_categorical_ids,
            own_numeric_features,
            own_worker_positions,
        )
        opponent_map = self.farm_encoder(
            opponent_categorical_ids,
            opponent_numeric_features,
            opponent_worker_positions,
        )

        own_summary = self.farm_pooling(own_map)
        opponent_summary = self.farm_pooling(opponent_map)
        global_summary = self.global_encoder(global_features)
        comparison = own_summary - opponent_summary

        state_vector = torch.cat(
            [
                own_summary,
                opponent_summary,
                comparison,
                global_summary,
            ],
            dim=1,
        )

        policy_context = torch.cat(
            [opponent_summary, global_summary],
            dim=1,
        )
        height, width = own_map.shape[-2:]
        policy_context = policy_context[:, :, None, None].expand(
            -1,
            -1,
            height,
            width,
        )
        policy_features = torch.cat(
            [own_map, policy_context],
            dim=1,
        )

        return StateEncoding(
            own_map=own_map,
            opponent_map=opponent_map,
            own_summary=own_summary,
            opponent_summary=opponent_summary,
            global_summary=global_summary,
            policy_features=policy_features,
            state_vector=state_vector,
        )


class UnitContextEncoder(nn.Module):
    """Aktif unit'in spatial konumu ile kendi inventory'sini birleştirir."""

    def __init__(
        self,
        spatial_channels: int = 192,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.spatial_channels = spatial_channels
        self.unit_feature_dim = len(UNIT_FEATURE_NAMES)
        self.output_dim = output_dim
        self.projection = nn.Sequential(
            nn.Linear(spatial_channels + self.unit_feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        policy_features: torch.Tensor,
        unit_positions_yx: torch.Tensor,
        unit_features: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Padded unit batch'ini ``(B, U, output_dim)`` context'e çevirir."""
        if policy_features.ndim != 4:
            raise ValueError("policy_features (B, C, H, W) olmalı.")
        if unit_positions_yx.ndim != 3 or unit_positions_yx.shape[-1] != 2:
            raise ValueError("unit_positions_yx (B, U, 2) olmalı.")
        if unit_features.ndim != 3:
            raise ValueError("unit_features (B, U, F) olmalı.")
        if unit_mask.shape != unit_positions_yx.shape[:2]:
            raise ValueError("unit_mask (B, U) olmalı.")
        if unit_features.shape[:2] != unit_positions_yx.shape[:2]:
            raise ValueError("Unit position ve feature boyutları eşleşmeli.")

        batch_size, _, height, width = policy_features.shape
        positions = unit_positions_yx.long()
        y = positions[..., 0]
        x = positions[..., 1]
        valid_positions = (
            (y >= 0)
            & (y < height)
            & (x >= 0)
            & (x < width)
            & unit_mask.bool()
        )
        if not torch.equal(valid_positions, unit_mask.bool()):
            raise ValueError("Geçerli unit pozisyonları farm sınırları içinde olmalı.")

        # Padded satırlar gather için geçici olarak (0, 0)'a yönlendirilir;
        # projection sonrası unit_mask ile tekrar sıfırlanır.
        safe_y = y.clamp(0, height - 1)
        safe_x = x.clamp(0, width - 1)
        batch_index = torch.arange(batch_size, device=policy_features.device)
        batch_index = batch_index[:, None].expand_as(safe_y)
        spatial_yx = policy_features.permute(0, 2, 3, 1)
        local_features = spatial_yx[batch_index, safe_y, safe_x]

        combined = torch.cat([local_features, unit_features], dim=-1)
        contexts = self.projection(combined)
        return contexts * unit_mask.unsqueeze(-1).to(contexts.dtype)


class CategorizedActionHeads(nn.Module):
    """Ortak hidden state'i gramer kategorisine özel logitlere dönüştürür."""

    LEARNABLE_HEAD_TYPES = (
        ActionHeadType.UNIT_OPERATION,
        ActionHeadType.MARKET_OPERATION,
        ActionHeadType.ITEM,
        ActionHeadType.QUANTITY,
    )

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                head_type.name.lower(): nn.Linear(
                    hidden_dim,
                    len(TOKENS_BY_HEAD[head_type]),
                )
                for head_type in self.LEARNABLE_HEAD_TYPES
            }
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if head_type_ids.shape != hidden_states.shape[:2]:
            raise ValueError("head_type_ids (B, T) olmalı.")
        if torch.any(head_type_ids < int(ActionHeadType.STRUCTURAL)) or torch.any(
            head_type_ids > int(ActionHeadType.QUANTITY)
        ):
            raise ValueError("Bilinmeyen action head type id kullanıldı.")

        # Global token arayüzünde her kategorinin local logitleri
        # yalnızca kendi global token kolonlarına scatter edilir. Grammar maskesi
        # aynı kategorideki observation'a özel geçersiz seçenekleri eleyebilir.
        global_logits = hidden_states.new_zeros(
            (*hidden_states.shape[:2], ACTION_VOCAB_SIZE)
        )
        for head_type in self.LEARNABLE_HEAD_TYPES:
            local_logits = self.heads[head_type.name.lower()](hidden_states)
            token_ids = torch.tensor(
                [TOKEN_TO_ID[token] for token in TOKENS_BY_HEAD[head_type]],
                dtype=torch.long,
                device=hidden_states.device,
            )
            scattered = hidden_states.new_zeros(
                (*hidden_states.shape[:2], ACTION_VOCAB_SIZE)
            ).scatter(
                dim=-1,
                index=token_ids.view(1, 1, -1).expand(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    -1,
                ),
                src=local_logits,
            )
            selector = (head_type_ids == int(head_type)).unsqueeze(-1)
            global_logits = torch.where(selector, scattered, global_logits)
        # STRUCTURAL satırlarda tüm logitler 0 kalır. Grammar tam olarak tek
        # tokenı açtığı için bu satırların log-probability'si daima 0'dır.
        return global_logits


class AutoregressiveActionDecoder(nn.Module):
    """State ve action prefix'inden bir sonraki token logitlerini üretir."""

    def __init__(
        self,
        state_dim: int = 256,
        unit_context_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.token_embedding = nn.Embedding(ACTION_VOCAB_SIZE, hidden_dim)
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.unit_projection = nn.Linear(unit_context_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_heads = CategorizedActionHeads(hidden_dim)

    @staticmethod
    def _gather_step_contexts(
        unit_contexts: torch.Tensor,
        context_unit_indices: torch.Tensor,
    ) -> torch.Tensor:
        if context_unit_indices.ndim != 2:
            raise ValueError("context_unit_indices (B, T) olmalı.")
        if unit_contexts.ndim != 3:
            raise ValueError("unit_contexts (B, U, D) olmalı.")
        if unit_contexts.shape[0] != context_unit_indices.shape[0]:
            raise ValueError("Context batch boyutları eşleşmeli.")

        unit_count = unit_contexts.shape[1]
        is_unit = context_unit_indices >= 0
        if torch.any(context_unit_indices[is_unit] >= unit_count):
            raise ValueError("Context içinde var olmayan bir unit index'i kullanıldı.")

        safe_indices = context_unit_indices.clamp(min=0)
        gathered = torch.gather(
            unit_contexts,
            dim=1,
            index=safe_indices.unsqueeze(-1).expand(
                -1,
                -1,
                unit_contexts.shape[-1],
            ),
        )
        return gathered * is_unit.unsqueeze(-1).to(gathered.dtype)

    def forward(
        self,
        state_vector: torch.Tensor,
        input_token_ids: torch.Tensor,
        context_unit_indices: torch.Tensor,
        head_type_ids: torch.Tensor,
        unit_contexts: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher forcing veya tek-adım generation için decoder forward'u.

        ``input_token_ids[:, t]`` işlenirken aynı satırdaki context,
        tahmin edilecek bir sonraki tokenın aktif unit'ine aittir.
        """
        if input_token_ids.dtype != torch.long:
            raise TypeError("Embedding girdisi input_token_ids torch.long olmalı.")
        if input_token_ids.shape != context_unit_indices.shape:
            raise ValueError("Token ve context sequence boyutları eşleşmeli.")
        if input_token_ids.shape != head_type_ids.shape:
            raise ValueError("Token ve head type sequence boyutları eşleşmeli.")

        token_vectors = self.token_embedding(input_token_ids)
        state_vectors = self.state_projection(state_vector).unsqueeze(1)
        step_contexts = self._gather_step_contexts(
            unit_contexts,
            context_unit_indices,
        )
        step_contexts = self.unit_projection(step_contexts)
        decoder_input = self.input_norm(
            token_vectors + state_vectors + step_contexts
        )
        hidden_states, next_hidden = self.gru(decoder_input, hidden)
        return self.output_heads(hidden_states, head_type_ids), next_hidden


class StateValueHead(nn.Module):
    def __init__(self, state_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        return self.network(state_vector).squeeze(-1)


class ActorCriticEncoding(NamedTuple):
    state: StateEncoding
    unit_contexts: torch.Tensor
    token_logits: torch.Tensor
    decoder_hidden: torch.Tensor
    value: torch.Tensor


class KaggricultureActorCritic(nn.Module):
    """Mevcut encoder, unit-context policy decoder ve scalar critic."""

    def __init__(self) -> None:
        super().__init__()
        self.state_encoder = StateEncoder()
        self.unit_context_encoder = UnitContextEncoder(
            spatial_channels=self.state_encoder.policy_channels,
            output_dim=128,
        )
        self.action_decoder = AutoregressiveActionDecoder(
            state_dim=self.state_encoder.state_dim,
            unit_context_dim=self.unit_context_encoder.output_dim,
        )
        self.value_head = StateValueHead(state_dim=self.state_encoder.state_dim)

    def encode_state_and_units(
        self,
        own_categorical_ids: torch.Tensor,
        own_numeric_features: torch.Tensor,
        own_worker_positions: torch.Tensor,
        opponent_categorical_ids: torch.Tensor,
        opponent_numeric_features: torch.Tensor,
        opponent_worker_positions: torch.Tensor,
        global_features: torch.Tensor,
        unit_positions_yx: torch.Tensor,
        unit_features: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[StateEncoding, torch.Tensor, torch.Tensor]:
        state = self.state_encoder(
            own_categorical_ids,
            own_numeric_features,
            own_worker_positions,
            opponent_categorical_ids,
            opponent_numeric_features,
            opponent_worker_positions,
            global_features,
        )
        unit_contexts = self.unit_context_encoder(
            state.policy_features,
            unit_positions_yx,
            unit_features,
            unit_mask,
        )
        value = self.value_head(state.state_vector)
        return state, unit_contexts, value

    def forward(
        self,
        own_categorical_ids: torch.Tensor,
        own_numeric_features: torch.Tensor,
        own_worker_positions: torch.Tensor,
        opponent_categorical_ids: torch.Tensor,
        opponent_numeric_features: torch.Tensor,
        opponent_worker_positions: torch.Tensor,
        global_features: torch.Tensor,
        unit_positions_yx: torch.Tensor,
        unit_features: torch.Tensor,
        unit_mask: torch.Tensor,
        input_token_ids: torch.Tensor,
        context_unit_indices: torch.Tensor,
        head_type_ids: torch.Tensor,
        decoder_hidden: torch.Tensor | None = None,
    ) -> ActorCriticEncoding:
        state, unit_contexts, value = self.encode_state_and_units(
            own_categorical_ids,
            own_numeric_features,
            own_worker_positions,
            opponent_categorical_ids,
            opponent_numeric_features,
            opponent_worker_positions,
            global_features,
            unit_positions_yx,
            unit_features,
            unit_mask,
        )
        token_logits, next_hidden = self.action_decoder(
            state.state_vector,
            input_token_ids,
            context_unit_indices,
            head_type_ids,
            unit_contexts,
            decoder_hidden,
        )
        return ActorCriticEncoding(
            state=state,
            unit_contexts=unit_contexts,
            token_logits=token_logits,
            decoder_hidden=next_hidden,
            value=value,
        )
