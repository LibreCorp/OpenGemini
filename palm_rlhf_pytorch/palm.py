
import math
import copy
from pathlib import Path
from collections import namedtuple
from functools import wraps
from itertools import zip_longest

from tqdm import tqdm
from beartype import beartype

import torch
from torch import einsum, nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, ModuleDict

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from palm_rlhf_pytorch.attention import Attention
from palm_rlhf_pytorch.utils import top_p, top_k, masked_mean, gumbel_sample, eval_decorator
from palm_rlhf_pytorch.lora import LoRA

from typing import Optional, Tuple

# -------------------------
# Router + MoE (FFN experts)
# -------------------------

class LoadBalancingLoss:
    @staticmethod
    def compute(router_probs):  # (b*n, E)
        # encourage uniform usage; KL to uniform
        E = router_probs.shape[-1]
        uniform = torch.full_like(router_probs, 1.0 / E)
        kl = F.kl_div((router_probs + 1e-9).log(), uniform, reduction='batchmean')
        return kl

class TopKRouter(Module):
    def __init__(self, dim, num_experts, k=2, capacity_factor=1.25, jitter_eps=1e-2):
        super().__init__()
        self.w = nn.Linear(dim, num_experts, bias=False)
        self.k = k
        self.capacity_factor = capacity_factor
        self.jitter_eps = jitter_eps
        self.num_experts = num_experts

    def forward(self, x):
        # x: (b, n, d) -> flatten tokens
        b, n, d = x.shape
        x_flat = x.reshape(b * n, d)
        logits = self.w(x_flat)
        if self.training and self.jitter_eps > 0:
            logits = logits + self.jitter_eps * torch.randn_like(logits)
        probs = logits.softmax(dim=-1)                          # (T, E)
        topk_val, topk_idx = probs.topk(self.k, dim=-1)         # (T, k)
        return probs, topk_val, topk_idx                         # router_probs for aux loss

class ExpertFFN(Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        inner = dim * hidden_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, inner * 2, bias=False),  # 2x for SwiGLU
            SwiGLU(),
            nn.Linear(inner, dim, bias=False)
        )
    def forward(self, x):
        return self.ff(x)

class MoE(Module):
    def __init__(self, dim, num_experts=16, k=2, capacity_factor=1.25, hidden_mult=4, aux_loss_weight=0.01):
        super().__init__()
        self.router = TopKRouter(dim, num_experts, k=k, capacity_factor=capacity_factor)
        self.experts = ModuleList([ExpertFFN(dim, hidden_mult) for _ in range(num_experts)])
        self.aux_loss_weight = aux_loss_weight
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor

    def forward(self, x):
        # x: (b, n, d)
        b, n, d = x.shape
        T = b * n
        router_probs, topk_val, topk_idx = self.router(x)

        # capacity per expert
        cap = math.ceil(self.capacity_factor * (T * self.k) / self.num_experts)

        # dispatch buffers
        device = x.device
        y = torch.zeros_like(x).view(T, d)
        # simple, memory-safe batched dispatch
        aux_loss = LoadBalancingLoss.compute(router_probs)

        for e in range(self.num_experts):
            # gather tokens routed to expert e (potentially up to capacity)
            mask_e = (topk_idx == e).any(dim=-1)      # tokens that chose e among their top-k
            idx_e = mask_e.nonzero(as_tuple=False).squeeze(-1)
            if idx_e.numel() == 0:
                continue
            if idx_e.numel() > cap:
                idx_e = idx_e[:cap]  # drop overflow (common MoE practice)

            x_e = x.view(T, d).index_select(0, idx_e)
            out_e = self.experts[e](x_e)
            # naive combine: average over multiple assignments if k>1
            y.index_add_(0, idx_e, out_e)

        y = y.view(b, n, d)
        return y, self.aux_loss_weight * aux_loss

# --------------------------------------
# Multimodal Adapters (native ingestion)
# --------------------------------------

class VisionAdapter(Module):
    def __init__(self, dim, vit_embed_dim=768, project=True):
        super().__init__()
        self.encoder = nn.Conv2d(3, vit_embed_dim, kernel_size=16, stride=16)  # patchify
        self.proj = nn.Linear(vit_embed_dim, dim) if project else nn.Identity()

    def forward(self, images):
        # images: (b, c=3, H, W)
        x = self.encoder(images)                     # (b, d_v, H/16, W/16)
        x = rearrange(x, 'b d h w -> b (h w) d')    # patch tokens
        x = self.proj(x)                             # -> LM dim
        return x

class AudioAdapter(Module):
    def __init__(self, dim, audio_embed_dim=512, project=True, n_mels=128):
        super().__init__()
        self.spec = torch.nn.Sequential(
            # TODO
        )
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, audio_embed_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(audio_embed_dim, audio_embed_dim, kernel_size=5, padding=2),
            nn.GELU()
        )
        self.proj = nn.Linear(audio_embed_dim, dim) if project else nn.Identity()

    def forward(self, mels):  # (b, n_mels, T)
        x = self.conv(mels)                                  # (b, audio_embed_dim, T)
        x = rearrange(x, 'b d t -> b t d')
        x = self.proj(x)
        return x

class VideoAdapter(Module):
    def __init__(self, dim, vit_embed_dim=768):
        super().__init__()
        self.vision = VisionAdapter(dim, vit_embed_dim)
        self.time_pos = nn.Parameter(torch.zeros(4096, dim))  # large enough for long clips

    def forward(self, frames):  # (b, t, 3, H, W)
        b, t, c, h, w = frames.shape
        frames = frames.view(b * t, c, h, w)
        toks = self.vision(frames)                            # (b*t, p, d)
        toks = toks.view(b, t, -1, toks.size(-1))             # (b, t, p, d)
        toks = toks.flatten(1, 2)                             # concat time+patches -> (b, t*p, d)
        # add time position embeddings (simple; replace with better temporal encoding if needed)
        pos = self.time_pos[:toks.size(1)]
        toks = toks + pos
        return toks

class LongContextConfig(namedtuple('LCC', ['chunk_size', 'use_chunked_attn', 'kv_cache'])):
    __slots__ = ()


def topk_distill_loss(student_logits, teacher_topk_idx, teacher_topk_probs, k):
    """
    Student matches teacher's k-sparse next-token distribution (2.5 report).
    teacher_topk_idx: (b, n, k); teacher_topk_probs: (b, n, k) normalized.
    """
    b, n, c = student_logits.shape
    # gather teacher mass at top-k indices
    # build sparse target in place (avoid dense vocab)
    logp = student_logits.log_softmax(dim=-1)
    idx = teacher_topk_idx
    tgt = teacher_topk_probs
    # NLL on sparse support
    gathered = logp.gather(-1, idx)                   # (b, n, k)
    loss = -(tgt * gathered).sum(dim=-1).mean()
    return loss

class ThinkingBudget:
    """
    Tracks auxiliary scratch tokens and budget accounting.
    Wire this into an RL loop (not shown) to learn when/how much to think.
    """
    def __init__(self, max_tokens: int = 0):
        self.max_tokens = max_tokens

    def enabled(self) -> bool:
        return self.max_tokens > 0

class ParallelTransformerBlockWithMoE(ParallelTransformerBlock):
    """
    Extends your block: swaps FFN with MoE when enabled, returns aux losses.
    """
    def __init__(self, *args, moe: Optional[MoE] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe = moe   # if None -> use dense FFN from parent

    def forward(self, x, mask=None, finetune_modules=None):
        # same attention path (y_attn) as parent; intercept FFN path
        n, device, h = x.shape[1], x.device, self.heads
        x_norm = self.norm(x)

        q, k, v, ff = self.fused_attn_ff_proj(x_norm).split(self.fused_dims, dim=-1)

        # optional LoRA
        if exists(finetune_modules):
            lora_q, lora_k, lora_v, lora_o = finetune_modules
            q = q + lora_q(x_norm)
            k = k + lora_k(x_norm)
            v = v + lora_v(x_norm)
        else:
            lora_o = None

        # heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        positions, scale = self.get_rotary_embedding(n, device)
        q = apply_rotary_pos_emb(positions, q, scale)
        k = apply_rotary_pos_emb(positions, k, scale ** -1)

        attn_out = self.attend(q, k, v, mask=mask)
        attn_out = rearrange(attn_out, "b h n d -> b n (h d)")
        attn_out = self.attn_out(attn_out)
        if exists(lora_o):
            attn_out = attn_out + lora_o(attn_out)

        # --- FFN path: MoE or dense
        aux = torch.tensor(0.0, device=device)
        if self.moe is not None:
            ff_out, aux = self.moe(ff.view(x.shape))   # ff was (b,n,dim*ff_mult*2) before SwiGLU
        else:
            ff_out = self.ff_out(ff)

        return attn_out + ff_out, aux

# ------------------------------
# GeminiAnalog top-level module
# ------------------------------

class GeminiAnalog(PaLM):
    @beartype
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        pro_tier: bool = True,
        num_experts: int = 16,
        topk_experts: int = 2,
        moe_capacity_factor: float = 1.25,
        moe_aux_loss_weight: float = 0.01,
        multimodal: bool = True,
        long_context: Optional[LongContextConfig] = None,
        thinking_budget_tokens: int = 0,
        **palm_kwargs
    ):
        super().__init__(dim=dim, num_tokens=num_tokens, depth=depth, **palm_kwargs)

        # replace blocks with MoE-capable versions
        new_layers = ModuleList([])
        moe = MoE(self.dim, num_experts=num_experts, k=topk_experts,
                  capacity_factor=moe_capacity_factor, hidden_mult=palm_kwargs.get('ff_mult', 4),
                  aux_loss_weight=moe_aux_loss_weight) if pro_tier else None

        self.pro_tier = pro_tier
        self.long_context = long_context or LongContextConfig(chunk_size=0, use_chunked_attn=False, kv_cache=True)
        self.thinking = ThinkingBudget(thinking_budget_tokens)

        for _ in range(depth):
            block = Residual(ParallelTransformerBlockWithMoE(
                dim=self.dim,
                causal=self.causal,
                dim_head=self.dim_head,
                heads=self.heads,
                qk_rmsnorm=palm_kwargs.get('qk_rmsnorm', False),
                ff_mult=palm_kwargs.get('ff_mult', 4),
                attn_dropout=palm_kwargs.get('attn_dropout', 0.),
                ff_dropout=palm_kwargs.get('ff_dropout', 0.),
                xpos_scale_base=palm_kwargs.get('rotary_xpos_scale_base', 512),
                flash_attn=palm_kwargs.get('flash_attn', False),
                moe=moe
            ))
            new_layers.append(block)
        self.layers = new_layers

        # multimodal adapters
        self.enable_multimodal = multimodal
        if multimodal:
            self.vision_adapter = VisionAdapter(dim=self.dim)
            self.audio_adapter  = AudioAdapter(dim=self.dim)
            self.video_adapter  = VideoAdapter(dim=self.dim)

        # distillation buffer (for Flash-tier or small models)
        self.register_buffer("_k_sparse_idx", None, persistent=False)
        self.register_buffer("_k_sparse_probs", None, persistent=False)
        self._distill_k = 0

    # ---- multimodal fusion ----
    def fuse_modalities(self, txt_ids=None, images=None, mels=None, frames=None):
        toks = []
        if txt_ids is not None:
            toks.append(self.token_emb(txt_ids))               # (b, n, d)
        if self.enable_multimodal and images is not None:
            vi = self.vision_adapter(images)                   # (b, p, d)
            toks.append(vi)
        if self.enable_multimodal and mels is not None:
            au = self.audio_adapter(mels)                      # (b, t, d)
            toks.append(au)
        if self.enable_multimodal and frames is not None:
            vd = self.video_adapter(frames)                    # (b, t*p, d)
            toks.append(vd)
        x = torch.cat(toks, dim=1) if len(toks) > 1 else toks[0]
        return x

    # ---- k-sparse distillation control ----
    def set_teacher_topk(self, idx: torch.Tensor, probs: torch.Tensor, k: int):
        """
        Provide teacher top-k indices & probs for k-sparse distill (2.5 paper).
        Shapes: (b, n, k).
        """
        self._k_sparse_idx = idx
        self._k_sparse_probs = probs
        self._distill_k = k

    # ---- forward (overrides a bit to return aux MoE loss & support modalities) ----
    def forward(
        self,
        x=None,
        *,
        return_loss=False,
        teacher_for_distill=False,
        disable_lora=False,
        finetune_scope=None,
        extra_embed=None,
        return_only_embedding=False,
        return_logits_with_embedding=False,
        images=None,      # (b, 3, H, W)
        mels=None,        # (b, n_mels, T)
        frames=None       # (b, t, 3, H, W)
    ):
        if x is not None and return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # inputs
        if x is not None:
            h = self.token_emb(x)
            if exists(extra_embed):
                h = h + extra_embed
            if self.enable_multimodal and any(v is not None for v in [images, mels, frames]):
                h = torch.cat([h, self.fuse_modalities(None, images, mels, frames)], dim=1)
        else:
            h = self.fuse_modalities(None, images, mels, frames)

        # LoRA finetune modules
        finetune_modules = tuple()
        if exists(finetune_scope) and not disable_lora:
            assert finetune_scope in self.finetune_modules
            finetune_modules = self.finetune_modules[finetune_scope]

        # run blocks, collect aux loss
        aux_total = torch.tensor(0.0, device=h.device)
        for layer, ft in zip_longest(self.layers, finetune_modules):
            out = layer(h, mask=None, finetune_modules=ft)
            if isinstance(out, tuple):
                h, aux = out
                aux_total = aux_total + aux
            else:
                h = out

        embeds = self.norm(h)
        if return_only_embedding:
            return embeds

        logits = self.to_logits(embeds)
        ret = (logits, embeds) if return_logits_with_embedding else logits

        # training losses
        if return_loss:
            ce = F.cross_entropy(rearrange(logits[:, :labels.size(1)], 'b n c -> b c n'), labels,
                                 ignore_index=self.cross_entropy_ignore_index)
            loss = ce + aux_total
            # optional k-sparse distill loss
            if self._distill_k > 0 and self._k_sparse_idx is not None and self._k_sparse_probs is not None:
                kd = topk_distill_loss(logits[:, :self._k_sparse_idx.size(1)], self._k_sparse_idx, self._k_sparse_probs, self._distill_k)
                loss = loss + kd
            return loss

        return ret

    # ---- generation with thinking budget (budget = extra internal steps before emitting visible tokens) ----
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        seq_len,
        prompt=None,
        temperature=1.,
        filter_logits_fn=top_k,
        filter_thres=0.9,
        pad_value=0,
        eos_token=None,
        use_tqdm=False,
        **kwargs
    ):
        # same as PaLM.generate, with optional useless-token burn for "thinking" (not output)
        # NOTE: This exposes the *mechanism*; proper RL to decide budget is separate.
        if not exists(prompt):
            prompt = torch.randint(0, self.num_tokens, (1, 1), device=self.device)

        out = prompt.clone()
        wrapper_fn = identity if not use_tqdm else tqdm

        # internal "thinking": roll forward max_tokens without appending to user-visible sequence
        if self.thinking.enabled():
            for _ in range(self.thinking.max_tokens):
                logits = super(GeminiAnalog, self).forward(out, **{**kwargs}).detach()  # forward-only internal
                _ = logits[:, -1]  # intentionally discard; this is compute budget

        # now generate visible tokens
        for _ in wrapper_fn(range(max(1, seq_len - prompt.shape[-1]))):
            logits = super(GeminiAnalog, self).forward(out, **{**kwargs})
            logits = logits[:, -1]
            if exists(filter_logits_fn):
                logits = filter_logits_fn(logits, thres=filter_thres)
            sample = gumbel_sample(logits, temperature=temperature, dim=-1)
            out, _ = pack([out, sample], 'b *')
            if exists(eos_token):
                is_eos = (out == eos_token)
                if is_eos.any(dim=-1).all():
                    shifted = F.pad(is_eos, (1, -1))
                    mask = shifted.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, pad_value)
                    break
        return out[:, prompt.shape[-1]:]
