"""Fused nucleus-membership reduction used only by cleanNSD.

No sorting, rank counting, entropy, TV, or vocabulary-sized mask buffers.
The comparison keeps the original strict-greater-logit and <= threshold rules.
CPU installations and installations without Triton use the reference reduction.
"""
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _prefix_mass(P, L, T, Partial, V: tl.constexpr, PS: tl.constexpr,
                     LS: tl.constexpr, TS: tl.constexpr, TILES: tl.constexpr,
                     BLOCK: tl.constexpr):
        row=tl.program_id(0); tile=tl.program_id(1)
        token=tl.load(T+row*TS)
        token_logit=tl.load(L+row*LS+token).to(tl.float32)
        indices=tile*BLOCK+tl.arange(0,BLOCK)
        logits=tl.load(L+row*LS+indices,mask=indices<V,other=-float('inf')).to(tl.float32)
        probs=tl.load(P+row*PS+indices,mask=indices<V,other=0).to(tl.float32)
        mass=tl.sum(tl.where(logits>token_logit,probs,0),0)
        tl.store(Partial+row*TILES+tile,mass)

    @triton.jit
    def _membership(Partial, Output, TILES: tl.constexpr, TAU: tl.constexpr,
                    BLOCK: tl.constexpr):
        row=tl.program_id(0); indices=tl.arange(0,BLOCK)
        partial=tl.load(Partial+row*TILES+indices,mask=indices<TILES,other=0)
        mass=tl.sum(partial,0)
        tl.store(Output+row,mass<=TAU)

def target_top_p_accept_mask(*, target_probs, target_logits, proposed_tokens,
                             acceptance_top_p):
    if (triton is None or not target_probs.is_cuda or target_probs.shape[0]!=1
            or target_probs.shape[-1]<4096 or target_probs.stride(-1)!=1
            or target_logits.stride(-1)!=1):
        token_logits=target_logits.gather(-1,proposed_tokens.unsqueeze(-1)).squeeze(-1)
        higher=target_logits>token_logits.unsqueeze(-1)
        return target_probs.masked_fill(~higher,0.).sum(-1)<=float(acceptance_top_p)
    rows=proposed_tokens.numel(); vocab=target_probs.shape[-1]
    tiles=triton.cdiv(vocab,4096)
    partial=torch.empty((rows,tiles),device=target_probs.device,dtype=torch.float32)
    result=torch.empty_like(proposed_tokens,dtype=torch.bool)
    _prefix_mass[(rows,tiles)](target_probs,target_logits,proposed_tokens,partial,
        vocab,target_probs.stride(1),target_logits.stride(1),proposed_tokens.stride(1),tiles,4096)
    _membership[(rows,)](partial,result,tiles,float(acceptance_top_p),triton.next_power_of_2(tiles))
    return result
