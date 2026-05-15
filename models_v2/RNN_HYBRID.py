"""
RNN_HYBRID - Phương pháp thứ 2 cho bài toán Multi-Modal DeepFake Detection (DGM4).
================================================================================

MỤC ĐÍCH
--------
File này định nghĩa một mô hình deepfake detection ĐA PHƯƠNG THỨC (ảnh + văn bản)
sử dụng các kỹ thuật DEEP LEARNING TRUYỀN THỐNG để xử lý văn bản, KHÔNG dùng
Transformer/BERT làm xương sống mã hóa ngữ nghĩa như HAMMER.

Đây được coi là PHƯƠNG PHÁP 2 (so sánh với HAMMER ở thư mục `models/`).
Mô hình này được đặt trong thư mục `models_v2/` để KHÔNG ảnh hưởng tới code gốc.

KIẾN TRÚC TỔNG QUAN (PHIÊN BẢN NÂNG CẤP)
----------------------------------------
Mô hình kết hợp nhiều khối RNN cổ điển (BiLSTM + GRU) cùng một Self-Attention
pooling nhằm khai thác tối đa thông tin ngữ cảnh hai chiều của câu, đồng thời
phối hợp với một backbone CNN (ResNet-18 pretrained) cho ảnh. Cuối cùng, ảnh
và văn bản được hợp nhất qua các khối Cross-Modal Multi-Head Attention xếp chồng.

PHIÊN BẢN MỚI NHẤT (text encoder cực mạnh + ResNet-50):
  * Text encoder xếp chồng: Embedding + PosEmbedding + N-gram CNN
      -> BiLSTM N lớp -> BiGRU M lớp -> Transformer Encoder K lớp
      -> Multi-head pool (attention + mean + max) -> CLS.
  * Image: ResNet-50 pretrained (thay cho ResNet-18 trước đây).
  * `d_hidden` mặc định 512.
  * MAC: MLP 2 lớp + `log_temp` để InfoNCE ổn định.
  * Khởi tạo nhỏ cho các head logit để không phá contrastive lúc đầu.

    ┌──────────────────────── TEXT BRANCH ────────────────────────┐
    | Embedding + PosEmb + N-gram CNN (k=2,3,4,5, residual)       |
    |   -> BiLSTM (n_bilstm_layers, BIG)                          |
    |   -> proj -> BiGRU (n_bigru_layers)                         |
    |   -> Transformer Encoder (n_self_attn_layers, n_heads)      |
    |   -> MultiHeadPool(attn + mean + max) -> CLS                |
    └─────────────────────────────────────────────────────────────┘

    ┌──────────────────────── IMAGE BRANCH ───────────────────────┐
    | ResNet-50 pretrained -> 1x1 conv proj -> patch tokens       |
    └─────────────────────────────────────────────────────────────┘

    ┌──────────────────── CROSS-MODAL FUSION ─────────────────────┐
    |   N x (text<->image MHA + FFN, LayerNorm pre-norm)          |
    └─────────────────────────────────────────────────────────────┘

ĐẦU RA (heads) khớp 1-1 với HAMMER để dùng chung train/val loop:
  - BIC : nhị phân thật/giả                       (CrossEntropy)
  - MLC : đa nhãn 4 lớp (FS, FA, TS, TA)          (BCEWithLogits)
  - bbox: hồi quy box vùng giả                    (L1 + GIoU)
  - TMG : phân loại token có bị manipulate        (CrossEntropy theo token)
  - MAC : InfoNCE đối xứng image-text             (projection MLP 2 lớp)

CONFIG (đọc từ dict `config`):
  hidden_dim        : kích thước ẩn chung (mặc định 384)
  n_bilstm_layers   : số lớp BiLSTM (mặc định 3)
  n_fusion_blocks   : số khối Cross-Modal (mặc định 3)
  n_heads           : số head trong MHA (mặc định 6)
  dropout           : dropout chung (mặc định 0.3)

INTERFACE
---------
Khớp với HAMMER để tái sử dụng cùng training/eval loop:

    # Train
    loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC = \
        model(image, label, text_input, fake_image_box, fake_token_pos, alpha=alpha)

    # Inference
    logits_real_fake, logits_multicls, output_coord, logits_tok = \
        model(image, label, text_input, fake_image_box, fake_token_pos, is_train=False)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

from ..models import box_ops
from ..tools.multilabel_metrics import get_multi_label


# ---------------------------------------------------------------------------
class _SelfAttnPool(nn.Module):
    """Self-attention pooling: học một vector query để gộp token features."""

    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim) * 0.02)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        scores = torch.einsum('bld,d->bl', x, self.q) / (x.size(-1) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        pooled = torch.einsum('bl,bld->bd', attn, x)
        return self.norm(self.proj(pooled))


# ---------------------------------------------------------------------------
class _NGramCNN(nn.Module):
    """Multi-scale 1D convolutions tren embedding de bat n-gram (2..5)."""

    def __init__(self, d_emb, d_out, kernels=(2, 3, 4, 5), dropout=0.1):
        super().__init__()
        assert d_out % len(kernels) == 0
        per = d_out // len(kernels)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_emb, per, kernel_size=k, padding=k // 2) for k in kernels
        ])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)
        x_t = x.transpose(1, 2)                  # (B, D, L)
        outs = []
        L = x.size(1)
        for conv in self.convs:
            y = conv(x_t)                        # (B, per, L'?)
            y = y[..., :L]                       # cat lai cho khop chieu
            outs.append(y)
        y = torch.cat(outs, dim=1).transpose(1, 2)   # (B, L, d_out)
        return self.drop(self.act(y))


class _MultiHeadPool(nn.Module):
    """Pool da goc nhin: attention + mean + max -> concat -> project."""

    def __init__(self, dim):
        super().__init__()
        self.attn = _SelfAttnPool(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, mask=None):
        # x: (B, L, D), mask: (B, L) 1=valid
        attn_vec = self.attn(x, mask)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            mean_vec = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
            x_for_max = x.masked_fill(m == 0, float('-inf'))
            max_vec = x_for_max.max(dim=1).values
        else:
            mean_vec = x.mean(dim=1)
            max_vec = x.max(dim=1).values
        return self.proj(torch.cat([attn_vec, mean_vec, max_vec], dim=-1))


class _TextRNNEncoder(nn.Module):
    """
    Text encoder ĐỘ PHỨC TẠP CAO NHẤT: kết hợp nhiều khối truyền thống.

    Pipeline:
      Embedding (d_emb)
        + learnable Positional Embedding
        + N-gram CNN (kernels 2/3/4/5) cộng vào embedding (residual)
      -> BiLSTM (n_bilstm_layers, hidden=d_hidden)            [2*d_hidden]
      -> Projection xuống d_hidden
      -> BiGRU (n_bigru_layers, hidden=d_hidden//2 each side) [d_hidden]
      -> Transformer Encoder (n_self_attn_layers, n_heads heads)
      -> LayerNorm + Dropout
      -> Multi-head pooling (attention + mean + max) -> CLS vector
    """

    def __init__(self, vocab_size, d_emb=512, d_hidden=512,
                 n_bilstm_layers=4, n_bigru_layers=2, n_self_attn_layers=2,
                 n_heads=8, max_len=160, pad_idx=0, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_emb, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_emb)
        self.emb_norm = nn.LayerNorm(d_emb)
        self.emb_drop = nn.Dropout(dropout)

        self.ngram_cnn = _NGramCNN(d_emb, d_emb, kernels=(2, 3, 4, 5), dropout=dropout)

        self.bilstm = nn.LSTM(
            input_size=d_emb,
            hidden_size=d_hidden,
            num_layers=n_bilstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_bilstm_layers > 1 else 0.0,
        )
        self.lstm_proj = nn.Linear(2 * d_hidden, d_hidden)
        self.lstm_norm = nn.LayerNorm(d_hidden)

        self.bigru = nn.GRU(
            input_size=d_hidden,
            hidden_size=d_hidden // 2,
            num_layers=n_bigru_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_bigru_layers > 1 else 0.0,
        )
        self.gru_norm = nn.LayerNorm(d_hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden, nhead=n_heads, dim_feedforward=d_hidden * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(enc_layer, num_layers=n_self_attn_layers)

        self.norm = nn.LayerNorm(d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.pool = _MultiHeadPool(d_hidden)
        self.out_dim = d_hidden
        self.max_len = max_len

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).clamp_max(self.max_len - 1)
        x = self.embedding(input_ids) + self.pos_emb(pos_ids).unsqueeze(0)
        x = self.emb_norm(x)
        x = x + self.ngram_cnn(x)                # residual n-gram
        x = self.emb_drop(x)

        x, _ = self.bilstm(x)                    # (B, L, 2D)
        x = self.lstm_norm(self.lstm_proj(x))    # (B, L, D)
        x, _ = self.bigru(x)                     # (B, L, D)
        x = self.gru_norm(x)

        # Transformer self-attention voi padding mask
        key_pad = (attention_mask == 0)
        x = self.self_attn(x, src_key_padding_mask=key_pad)

        x = self.dropout(self.norm(x))
        cls = self.pool(x, attention_mask)
        return x, cls


# ---------------------------------------------------------------------------
class _ImageCNNEncoder(nn.Module):
    """ResNet-50 pretrained -> patch tokens projected to common dim."""

    def __init__(self, out_dim=512, pretrained=True):
        super().__init__()
        try:
            weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = torchvision.models.resnet50(weights=weights)
        except Exception:
            backbone = torchvision.models.resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        # ResNet50 layer4 output: 2048 channels
        self.proj = nn.Conv2d(2048, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, image):
        feat = self.stem(image)
        feat = self.proj(feat)
        B, D, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)        # (B, HW, D)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return tokens, pooled, (H, W)


# ---------------------------------------------------------------------------
class _CrossModalBlock(nn.Module):
    """Pre-norm two-way cross attention between text tokens and image patches."""

    def __init__(self, dim, heads=6, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.ln_t1 = nn.LayerNorm(dim)
        self.ln_v1 = nn.LayerNorm(dim)
        self.t2v = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.v2t = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln_t2 = nn.LayerNorm(dim)
        self.ln_v2 = nn.LayerNorm(dim)
        self.ffn_t = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.ffn_v = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, t_tok, v_tok, t_mask=None):
        key_pad = (t_mask == 0) if t_mask is not None else None
        t_n = self.ln_t1(t_tok)
        v_n = self.ln_v1(v_tok)
        t_new, _ = self.t2v(query=t_n, key=v_n, value=v_n)
        v_new, _ = self.v2t(query=v_n, key=t_n, value=t_n, key_padding_mask=key_pad)
        t_tok = t_tok + t_new
        v_tok = v_tok + v_new
        t_tok = t_tok + self.ffn_t(self.ln_t2(t_tok))
        v_tok = v_tok + self.ffn_v(self.ln_v2(v_tok))
        return t_tok, v_tok


# ---------------------------------------------------------------------------
class _ProjectionHead(nn.Module):
    """MLP 2-lop cho contrastive (chinh + chuan hoa de InfoNCE on dinh)."""

    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
class RNN_HYBRID(nn.Module):
    """Mô hình lai RNN + CNN + Cross-Modal Attention. Tương thích interface HAMMER."""

    def __init__(self, args=None, config=None, text_encoder='bert-base-uncased',
                 tokenizer=None, init_deit=True):
        super().__init__()
        cfg = config or {}
        d_hidden = int(cfg.get('hidden_dim', 512))
        n_bilstm = int(cfg.get('n_bilstm_layers', 4))
        n_bigru = int(cfg.get('n_bigru_layers', 2))
        n_self_attn = int(cfg.get('n_self_attn_layers', 2))
        n_fusion = int(cfg.get('n_fusion_blocks', 4))
        n_heads = int(cfg.get('n_heads', 8))
        dropout = float(cfg.get('dropout', 0.3))

        self.config = cfg
        self.d_hidden = d_hidden

        vocab_size = tokenizer.vocab_size if tokenizer is not None else 30522
        pad_idx = tokenizer.pad_token_id if tokenizer is not None else 0

        self.text_encoder = _TextRNNEncoder(
            vocab_size=vocab_size, d_emb=d_hidden, d_hidden=d_hidden,
            n_bilstm_layers=n_bilstm, n_bigru_layers=n_bigru,
            n_self_attn_layers=n_self_attn, n_heads=n_heads,
            pad_idx=pad_idx, dropout=dropout,
        )
        self.image_encoder = _ImageCNNEncoder(out_dim=d_hidden, pretrained=init_deit)

        self.fusion = nn.ModuleList([
            _CrossModalBlock(d_hidden, heads=n_heads, dropout=0.1)
            for _ in range(n_fusion)
        ])

        # Heads -- khoi tao nho de logit khong qua lon luc dau
        self.bic_head = nn.Sequential(
            nn.LayerNorm(d_hidden * 2),
            nn.Linear(d_hidden * 2, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 2),
        )
        self.mlc_head = nn.Sequential(
            nn.LayerNorm(d_hidden * 2),
            nn.Linear(d_hidden * 2, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 4),
        )
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 4), nn.Sigmoid(),
        )
        self.tmg_head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 2),
        )

        # Projection cho contrastive (MAC) - MLP 2 lop
        self.proj_t = _ProjectionHead(d_hidden, d_hidden)
        self.proj_v = _ProjectionHead(d_hidden, d_hidden)
        # log_temp: temp = exp(log_temp). Khoi tao log(0.07) ~ -2.659
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.07)))

        self._init_small_heads()

    # ------------------------------------------------------------------
    def _init_small_heads(self):
        for m in [self.bic_head[-1], self.mlc_head[-1], self.tmg_head[-1]]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _encode(self, image, text_input):
        t_tok, t_cls = self.text_encoder(text_input.input_ids, text_input.attention_mask)
        v_tok, v_pool, _ = self.image_encoder(image)

        for blk in self.fusion:
            t_tok, v_tok = blk(t_tok, v_tok, t_mask=text_input.attention_mask)

        mask = text_input.attention_mask.unsqueeze(-1).float()
        t_pool_fused = (t_tok * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        v_pool_fused = v_tok.mean(dim=1)

        return t_tok, v_tok, t_pool_fused, v_pool_fused, t_cls, v_pool

    # ------------------------------------------------------------------
    def forward(self, image, label, text_input, fake_image_box, fake_token_pos,
                alpha=0.0, is_train=True):

        t_tok, v_tok, t_pool, v_pool, t_cls_raw, v_pool_raw = self._encode(image, text_input)
        fused = torch.cat([t_pool, v_pool], dim=-1)

        logits_real_fake = self.bic_head(fused)
        logits_multicls = self.mlc_head(fused)
        output_coord = self.bbox_head(v_pool)
        logits_tok = self.tmg_head(t_tok)

        if not is_train:
            return logits_real_fake, logits_multicls, output_coord, logits_tok

        device = image.device

        # ---------- BIC ----------
        cls_label = torch.ones(len(label), dtype=torch.long, device=device)
        real_pos = np.where(np.array(label) == 'orig')[0].tolist()
        cls_label[real_pos] = 0
        loss_BIC = F.cross_entropy(logits_real_fake, cls_label)

        # ---------- MLC ----------
        multi_label, _ = get_multi_label(label, image)
        loss_MLC = F.binary_cross_entropy_with_logits(
            logits_multicls, multi_label.float()
        )

        # ---------- bbox / giou ----------
        valid_box_mask = fake_image_box.sum(dim=-1) > 0
        if valid_box_mask.any():
            pred_b = output_coord[valid_box_mask]
            gt_b = fake_image_box[valid_box_mask].to(device)
            loss_bbox = F.l1_loss(pred_b, gt_b)
            giou = box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(pred_b),
                box_ops.box_cxcywh_to_xyxy(gt_b),
            )
            loss_giou = (1 - torch.diag(giou)).mean()
        else:
            loss_bbox = torch.zeros((), device=device)
            loss_giou = torch.zeros((), device=device)

        # ---------- TMG ----------
        token_label = text_input.attention_mask[:, 1:].clone()
        token_label[token_label == 0] = -100
        token_label[token_label == 1] = 0
        for b_idx, positions in enumerate(fake_token_pos):
            for pos in positions:
                if pos < token_label.size(1):
                    token_label[b_idx, pos] = 1
        logits_tok_aligned = logits_tok[:, 1:, :]
        L = min(logits_tok_aligned.size(1), token_label.size(1))
        loss_TMG = F.cross_entropy(
            logits_tok_aligned[:, :L, :].reshape(-1, 2),
            token_label[:, :L].reshape(-1),
            ignore_index=-100,
        )

        # ---------- MAC (InfoNCE doi xung) ----------
        zt = F.normalize(self.proj_t(t_cls_raw), dim=-1)
        zv = F.normalize(self.proj_v(v_pool_raw), dim=-1)
        temp = self.log_temp.exp().clamp(min=1e-3, max=1.0)
        logits_tv = zt @ zv.t() / temp
        targets = torch.arange(zt.size(0), device=device)
        loss_MAC = 0.5 * (
            F.cross_entropy(logits_tv, targets)
            + F.cross_entropy(logits_tv.t(), targets)
        )

        return loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC
