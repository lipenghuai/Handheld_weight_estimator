import torch
import torch.nn as nn

from .pointnet_encoder import PointNetEncoder
from .dgcnn_encoder import DGCNNEncoder
from .folding_decoder import FoldingDecoder


class PointCloudAE(nn.Module):
    """
    一键导入的点云 AutoEncoder：
      - encoder(points) -> z
      - decoder(z) -> recon_points
    """
    def __init__(
        self,
        n_points: int = 2048,
        latent_dim: int = 256,
        width_mult: float = 1.0,
        enc_dropout: float = 0.0,
        use_bn: bool = True,
        encoder_type: str = "dgcnn",   # "pointnet" 或 "dgcnn"
        dgcnn_k: int = 20,
        dgcnn_pool: str = "max+avg",
    ):
        super().__init__()
        self.n_points = n_points
        self.latent_dim = latent_dim
        self.width_mult = width_mult
        self.encoder_type = encoder_type

        if encoder_type == "pointnet":
            self.encoder = PointNetEncoder(
                latent_dim=latent_dim,
                width_mult=width_mult,
                dropout=enc_dropout,
                use_bn=use_bn,
            )
        elif encoder_type == "dgcnn":
            self.encoder = DGCNNEncoder(
                latent_dim=latent_dim,
                k=dgcnn_k,
                width_mult=width_mult,
                dropout=enc_dropout,
                use_bn=use_bn,
                global_pool=dgcnn_pool,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.decoder = FoldingDecoder(
            latent_dim=latent_dim,
            n_points=n_points,
            width_mult=width_mult,
            use_bn=use_bn,
        )

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        return self.encoder(points)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, points: torch.Tensor):
        z = self.encode(points)
        recon = self.decode(z)
        return recon, z

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
