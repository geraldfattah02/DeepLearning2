from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 256,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # input dim: left track (n_track, 2) + right track (n_track, 2)
        input_dim = 2 * n_track * 2
        output_dim = n_waypoints * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.shape[0]
        # Flatten and concatenate
        x = torch.cat([track_left.reshape(b, -1), track_right.reshape(b, -1)], dim=-1)
        out = self.net(x)
        out = out.view(b, self.n_waypoints, 2)
        return out.view(b, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 128,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Project 2D track points into d_model
        self.input_proj = nn.Linear(2, d_model)

        # positional embedding for track sequence (optional, but helpful)
        self.pos_embed = nn.Parameter(torch.randn(2 * n_track, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # output projection from decoder embeddings to 2D waypoints
        self.out_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.shape[0]
        device = track_left.device

        # concatenate left/right into a sequence (b, seq_len, 2)
        seq = torch.cat([track_left, track_right], dim=1)
        seq_len = seq.shape[1]  # should be 2 * n_track

        # project to d_model
        memory = self.input_proj(seq)  # (b, seq_len, d_model)
        # add positional embedding
        pos = self.pos_embed[:seq_len].unsqueeze(0).to(device)  # (1, seq_len, d_model)
        memory = memory + pos

        # transformer expects (S, B, E)
        memory = memory.permute(1, 0, 2).contiguous()

        # prepare target queries: (T, B, E)
        query_ids = torch.arange(self.n_waypoints, device=device)
        tgt = self.query_embed(query_ids)  # (T, E)
        tgt = tgt.unsqueeze(1).repeat(1, b, 1)

        # decode
        out = self.decoder(tgt=tgt, memory=memory)  # (T, B, E)

        # project to 2D
        out = out.permute(1, 0, 2).contiguous()  # (B, T, E)
        out = self.out_proj(out)  # (B, T, 2)

        return out


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # simple conv backbone to process 96x128 input
        # output spatial dims after convs will be small; then global pool + fc
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n_waypoints, 2)
        """
        x = image
        
        # Skip normalization as requested
        # x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Forward through CNN backbone
        x = self.backbone(x)
        
        # Global average pooling to get (B, 512, 1, 1)
        x = self.pool(x)
        
        # Flatten to (B, 512)
        x = x.view(x.size(0), -1)
        
        # Forward through fully connected layers
        x = self.fc(x)
        
        # Reshape from (B, n_waypoints * 2) to (B, n_waypoints, 2)
        x = x.view(-1, self.n_waypoints, 2)
        
        return x
        


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
