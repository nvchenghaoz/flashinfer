"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SM120/SM121 MXFP8 x MXFP4 MoE: two-kernel FC1 -> FC2 chain vs the fused
FC1+FC2 kernel, in one process on one GPU (``SM12xMxfp8Mxfp4Config(fused_fc12=...)``).

Each point times the whole runner forward (output zero-fill, route + MXFP8
quantize, FC1, FC2) under a CUDA graph with a cold L2, so the launch gap the
fused kernel removes is counted. Shapes are one rank's local experts; for an
expert-parallel shard, ids are drawn over ``--global-experts`` and those owned
by another rank are passed as -1 (unrouted), as the runner expects.

Usage:
    python benchmarks/bench_sm12x_mxfp8_mxfp4_fused_fc12.py
    python benchmarks/bench_sm12x_mxfp8_mxfp4_fused_fc12.py --shapes dsv41_flash_ep2tp2 --tokens 1 8 32
"""

import argparse

import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SM12xMxfp8Mxfp4Config,
    SwiGLU,
)
from flashinfer.testing.utils import bench_gpu_time

# name: (hidden, intermediate per rank, local experts, global experts, top-k)
SHAPES = {
    # DeepSeek-V4.1-Flash on one moe_tp2 x moe_ep2 rank: 384 experts / EP2,
    # intermediate 2304 / TP2.
    "dsv41_flash_ep2tp2": (4096, 1152, 192, 384, 6),
    # Qwen3-30B-A3B-like.
    "qwen3_moe_30b": (2048, 768, 128, 128, 8),
    # gpt-oss-120b-like (hidden and intermediate 2880 padded to 3072).
    "gpt_oss_like": (3072, 3072, 128, 128, 4),
}
TOKENS = (1, 2, 4, 8, 16, 32, 64, 128)


def make_routing(tokens, local_experts, global_experts, top_k, pattern, seed=0):
    """``uniform``: top-k distinct experts uniformly at random. ``skewed``: a
    Zipf-like popularity (a few hot experts take most rows)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(tokens, global_experts, generator=gen, device="cuda")
    if pattern == "skewed":
        rank = torch.arange(global_experts, device="cuda", dtype=torch.float32)
        scores = scores + 4.0 / (1.0 + rank / 4.0)
    ids = scores.topk(top_k, dim=1).indices.to(torch.int32)
    ids = torch.where(ids < local_experts, ids, torch.full_like(ids, -1))
    weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=1)
    return ids, weights


def make_weights(hidden, intermediate, experts):
    """Random E2M1 codes with unit UE8M0 scales (127): timing does not depend
    on the values, and this avoids quantizing BF16 weights."""
    u8 = dict(dtype=torch.uint8, device="cuda")
    return SM12xMxfp8Mxfp4Config.prepare_weights(
        torch.randint(0, 256, (experts, 2 * intermediate, hidden // 2), **u8),
        torch.full((experts, hidden // 128, 8 * intermediate), 127, **u8),
        torch.randint(0, 256, (experts, hidden, intermediate // 2), **u8),
        torch.full((experts, intermediate // 128, 4 * hidden), 127, **u8),
    )


def make_runner(view, hidden, intermediate, experts, top_k, fused):
    backend = SM12xMxfp8Mxfp4Config(fused_fc12=fused)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=experts, top_k=top_k),
        quant=QuantConfig(
            weight=QuantFormat.MXFP4,
            activation=QuantFormat.MXFP8,
            per_token_scale=False,
        ),
        experts=ExpertConfig(intermediate_size=intermediate, local_num_experts=experts),
        activation=SwiGLU(),
        backend=BackendOptions(candidates=(backend,)),
        execution=ExecutionConfig(enable_pdl=True),
    )
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for("sm12x_mxfp8_mxfp4", view)
    return MoELayer(config, device=torch.device("cuda")).runners[0], weight_pack


def time_us(runner, inputs, repeat_ms):
    times = bench_gpu_time(
        lambda: runner.forward(inputs),
        dry_run_time_ms=25,
        repeat_time_ms=repeat_ms,
        use_cuda_graph=True,
        enable_cupti=False,
        cold_l2_cache=True,
    )
    return 1e3 * float(torch.tensor(times).median())


def run(shape_names, tokens_list, patterns, variants, repeat_ms=100):
    """``variants``: name -> (fused_fc12, optional fc1 override). Prints one
    row per point and returns {(shape, pattern, tokens): {variant: us}}."""
    results = {}
    for name in shape_names:
        hidden, inter, experts, global_experts, top_k = SHAPES[name]
        view = make_weights(hidden, inter, experts)
        runners = {}
        for vname, (fused, fc1) in variants.items():
            runner, weight_pack = make_runner(
                view, hidden, inter, experts, top_k, fused
            )
            if fc1 is not None:
                runner._fc1 = fc1
            runners[vname] = runner
        for pattern in patterns:
            for tokens in tokens_list:
                ids, weights = make_routing(
                    tokens, experts, global_experts, top_k, pattern
                )
                x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
                act = MoEActivationPack(
                    hidden_states_q=x,
                    hidden_states_scale=None,
                    topk_ids=ids,
                    topk_weights=weights,
                )
                row, outs = {}, {}
                for vname, runner in runners.items():
                    inputs = runner.pack_inputs(act, weight_pack)
                    outs[vname] = runner.forward(inputs).float().clone()
                    row[vname] = time_us(runner, inputs, repeat_ms)
                ref = next(iter(outs.values()))
                err = max(
                    ((o - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
                    for o in outs.values()
                )
                results[name, pattern, tokens] = row
                cells = "  ".join(f"{v}={t:8.2f}" for v, t in row.items())
                print(
                    f"{name:20s} {pattern:7s} M={tokens:4d}  {cells}  "
                    f"max_rel_diff={err:.1e}",
                    flush=True,
                )
        del view, runners
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=SHAPES)
    parser.add_argument("--tokens", nargs="+", type=int, default=list(TOKENS))
    parser.add_argument(
        "--routing",
        nargs="+",
        default=["uniform", "skewed"],
        choices=["uniform", "skewed"],
    )
    parser.add_argument("--repeat-ms", type=int, default=100)
    args = parser.parse_args()
    print(f"GPU: {torch.cuda.get_device_name()}  times: median us per MoE forward")
    run(
        args.shapes,
        args.tokens,
        args.routing,
        {"chain": (False, None), "fused": (True, None), "auto": (None, None)},
        args.repeat_ms,
    )


if __name__ == "__main__":
    main()
