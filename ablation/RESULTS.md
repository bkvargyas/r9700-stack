# Kernel ablation: tcclaviger/vllm:dev, Flash-Next GPTQ, 2x R9700 TP2 (2026-09-18)

Standing config (P2P, auto expert offload, cudagraph [4], MTP-3), compile cache disabled per run, bench/bench.py.
Noise: ~1-2% single, ~5% aggregate. Checks 5/6 everywhere (the one miss is a model spelling weakness, same in base).

| variant | toggles | single | @4 | @8 | @16 | pf 2k | pf 8k |
|---|---|---|---|---|---|---|---|
| base | `none` | 82.5 | 81.1 | 146.4 | 145.6 | 2542 | 4131 |
| no_clav_attn | `CLAV_ATTN=0` | 83.4 (+1%) | 77.2 (-5%) | 144.1 (-2%) | 144.4 (-1%) | 2536 (-0%) | 4099 (-1%) |
| no_gdn_hip | `NO_AMD_GDN_HIP=1` | 81.7 (-1%) | 71.9 (-11%) | 136.8 (-7%) | 132.5 (-9%) | 2678 (+5%) | 4046 (-2%) |
| no_clav_helpers | `CLAV_HC=0 CLAV_CONV1D=0 CLAV_PLECONV=0 CLAV_SILU_QUANT=0 CLAV_MEMCPY=0 CLAV_STATE_COPY=0 CLAV_RESHAPE_CACHE=0` | 81.3 (-1%) | 80.8 (-0%) | 131.7 (-10%) | 148.9 (+2%) | 2543 (+0%) | 3976 (-4%) |
| no_fp8hip | `VLLM_DISABLE_FP8HIP=1` | 77 (-7%) | 73.4 (-9%) | 127.3 (-13%) | 146.7 (+1%) | 2685 (+6%) | 3885 (-6%) |
| no_rdna4_fp8 | `VLLM_DISABLE_RDNA4_FP8_KERNEL=1` | 82.6 (+0%) | 84.5 (+4%) | 148 (+1%) | 141.8 (-3%) | 2707 (+6%) | 4112 (-0%) |
| no_r4d_qsa | `R4D_QSA=0 R4D_QSA_PREP=0` | 78.1 (-5%) | 70.4 (-13%) | 130.2 (-11%) | 126.4 (-13%) | 2514 (-1%) | 4095 (-1%) |
| no_r4d_ple | `R4D_PLE=0` | 78.2 (-5%) | 80.4 (-1%) | 145.1 (-1%) | 147 (+1%) | 2654 (+4%) | 4115 (-0%) |
| no_r4d_select | `R4D_SELECT=0` | 81.4 (-1%) | 78.8 (-3%) | 144.5 (-1%) | 144.7 (-1%) | 2684 (+6%) | 4067 (-2%) |
| no_ar_quant | `R4D_AR_QUANT=0` | 82.6 (+0%) | 77.8 (-4%) | 146.1 (-0%) | 139.4 (-4%) | 2574 (+1%) | 3757 (-9%) |
| no_r4d_mxfp4 | `VLLM_DISABLE_R4D_MXFP4=1` | DIED (expert offload needs r4d MoE) | | | | | |
| no_all_clav | `DISABLE_ALL_CLAV=1` | DIED (expert offload needs r4d MoE) | | | | | |
