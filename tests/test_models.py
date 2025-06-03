import torch
from src.models.cnn_encoder import CNNEncoder
from src.models.transformer_encoder import TransformerEncoder
from src.models.fusion_net import FusionNet


def test_cnn_encoder_output_dim():
    B, T, F = 4, 200, 7
    x = torch.randn(B, T, F)
    model = CNNEncoder(in_channels=F, n_classes=5, latent_dim=16)
    lat = model(x)
    assert lat.shape == (B, 16)
    assert model.get_output_dim() == 16
    lat2, logits = model(x, return_logits=True)
    assert lat2.shape == (B, 16)
    assert logits.shape == (B, 5)


def test_transformer_encoder_output_dim():
    B, T, F = 3, 200, 20
    x = torch.randn(B, T, F)
    model = TransformerEncoder(in_channels=F, n_classes=5, latent_dim=32, n_layers=1, n_heads=4)
    lat = model(x)
    assert lat.shape == (B, 32)
    assert model.get_output_dim() == 32
    lat2, logits = model(x, return_logits=True)
    assert lat2.shape == (B, 32)
    assert logits.shape == (B, 5)


def test_fusion_shapes_concat_and_gated():
    B, T, F = 2, 200, 10
    seq = torch.randn(B, T, F)
    sym = torch.randn(B, 6)
    enc = CNNEncoder(in_channels=F, n_classes=5, latent_dim=8)

    concat = FusionNet(enc, sym_dim=6, n_classes=5, fusion_type="concat")
    fused, logits = concat(seq, sym, return_latent=True)
    assert fused.shape == (B, enc.get_output_dim() + 6)
    assert logits.shape == (B, 5)

    gated = FusionNet(enc, sym_dim=6, n_classes=5, fusion_type="gated")
    fused_g, logits_g = gated(seq, sym, return_latent=True)
    assert fused_g.shape == (B, enc.get_output_dim())
    assert logits_g.shape == (B, 5)


def test_meta_fusion_average():
    B, T, F = 2, 200, 10
    seq = torch.randn(B, T, F)
    sym = torch.randn(B, 6)
    enc1 = CNNEncoder(in_channels=F, n_classes=5, latent_dim=8)
    enc2 = CNNEncoder(in_channels=F, n_classes=5, latent_dim=8)
    model1 = FusionNet(enc1, sym_dim=6, n_classes=5)
    model2 = FusionNet(enc2, sym_dim=6, n_classes=5)
    logits1 = model1(seq, sym)
    logits2 = model2(seq, sym)
    meta = torch.stack([logits1, logits2]).mean(dim=0)
    assert meta.shape == (B, 5)

