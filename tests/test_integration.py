"""The plugin's vLLM integration points, against the installed (stock) vLLM. Run inside the image with the plugin
pip-installed (entry points), one GPU visible. Fails loudly when a vLLM pin bump moves something we rely on."""
import os
import sys

import torch

ok = True


def check(what, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {what}{(' -- ' + str(detail)) if detail and not cond else ''}")


# --- entry points + platform ------------------------------------------------------------------------------
from importlib.metadata import entry_points
eps = {g: {e.name: e.value for e in entry_points(group=g)} for g in ("vllm.general_plugins", "vllm.platform_plugins")}
check("general_plugins entry point", eps["vllm.general_plugins"].get("r9700") == "r9700_vllm:register", eps)
check("platform_plugins entry point", eps["vllm.platform_plugins"].get("r9700") == "r9700_vllm.platform:detect", eps)

from vllm.platforms import current_platform
from r9700_vllm.platform import R9700Platform
check("current_platform is R9700Platform", isinstance(current_platform, R9700Platform), type(current_platform))
check("is_rocm() still true", current_platform.is_rocm())
check("communicator -> R9kCommunicator",
      current_platform.get_device_communicator_cls() == "r9700_vllm.comm.r4d_ar.R9kCommunicator")
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
comm = resolve_obj_by_qualname(current_platform.get_device_communicator_cls())
check("R9kCommunicator subclasses stock CudaCommunicator", issubclass(comm, CudaCommunicator))

# --- general plugin -----------------------------------------------------------------------------------------
from vllm.plugins import load_general_plugins
load_general_plugins()

from vllm.model_executor.layers.quantization import get_quantization_config
from r9700_vllm.quant.ct import R9kCompressedTensorsConfig, R9kMxfp4MoEMethod
check("'compressed-tensors' -> R9kCompressedTensorsConfig",
      get_quantization_config("compressed-tensors") is R9kCompressedTensorsConfig)

from vllm import ModelRegistry
from r9700_vllm.models import qwen4_exp as Q
for arch, qual in Q.ARCHS.items():
    cls = ModelRegistry._try_load_model_cls(arch)
    want = getattr(Q, qual.split(":")[1])
    check(f"registry {arch} -> {want.__name__}", cls is want, cls)

# --- quant config fix-ups -------------------------------------------------------------------------------------
cfg = {
    "quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized",
    "config_groups": {
        "g_mxfp4": {"targets": ["re:.*experts.*"], "weights": {"num_bits": 4, "type": "float", "group_size": 32,
                                                              "strategy": "group", "symmetric": True,
                                                              "scale_dtype": "torch.uint8"}},
        "g_fp8": {"targets": ["re:.*self_attn.*"], "weights": {"num_bits": 8, "type": "float", "strategy": "block",
                                                              "block_structure": [128, 128], "symmetric": True}},
        "g_mtp": {"targets": ["re:mtp.*mlp.*"], "weights": {"num_bits": 8, "type": "float", "strategy": "block",
                                                            "block_structure": [128, 128], "symmetric": True}},
    },
    "ignore": ["lm_head"],
}
import copy
c = copy.deepcopy(cfg)
try:
    R9kCompressedTensorsConfig.from_config(c)
    check("from_config accepts a fork-style config", True)
except Exception as e:
    check("from_config accepts a fork-style config", False, e)
check("format-less fp8 group -> float-quantized", c["config_groups"]["g_fp8"].get("format") == "float-quantized")
check("MTP MLP fp8 group dropped (R9K_MTP_MLP=mxfp4)", "g_mtp" not in c["config_groups"])

# --- checkpoint stream filters ----------------------------------------------------------------------------------
gen, _ = Q.target_weights([
    ("model.language_model.layers.3.self_attn.q_scale", 0), ("model.language_model.layers.3.self_attn.attn.k_scale", 0),
    ("model.language_model.layers.3.self_attn.qkv_proj.weight_scale_inv", 1), ("lm_head.weight", 2)])
names = [n for n, _ in gen]
check("target filter", names == ["model.language_model.layers.3.self_attn.qkv_proj.weight_scale", "lm_head.weight"],
      names)
w = (torch.randn(640, 256) * 0.1).to(torch.float8_e4m3fn)
s = torch.rand(5, 2) + 0.5
mt = list(Q.mtp_weights([("mtp.lm_head.weight_q4", 0),
                         ("mtp.layers.0.mlp.experts.7.up_proj.weight", w),
                         ("mtp.layers.0.mlp.experts.7.up_proj.weight_scale_inv", s),
                         ("mtp.layers.0.self_attn.o_proj.weight_scale_inv", 3)], True))
mn = {n: t for n, t in mt}
check("MTP filter: fork head dropped, fp8 expert -> MXFP4, scale renamed",
      set(mn) == {"mtp.layers.0.mlp.experts.7.up_proj.weight_packed", "mtp.layers.0.mlp.experts.7.up_proj.weight_scale",
                  "mtp.layers.0.self_attn.o_proj.weight_scale"}, list(mn))
p = mn.get("mtp.layers.0.mlp.experts.7.up_proj.weight_packed")
check("MTP MXFP4 shapes", p is not None and tuple(p.shape) == (640, 128)
      and tuple(mn["mtp.layers.0.mlp.experts.7.up_proj.weight_scale"].shape) == (640, 8))

# --- construction scope restores what it touches ------------------------------------------------------------
from vllm.models.qwen4_exp.amd import ple_layer
stock = ple_layer.PLEVocabParallelEmbedding
pin = torch.Tensor.pin_memory
os.environ["R9K_PLE_FORMAT"] = "int6"
with Q.construction_scope():
    inside = ple_layer.PLEVocabParallelEmbedding
    t = torch.empty(3, 5).pin_memory()
check("scope installs int6 PLE subclass", inside is not stock and issubclass(inside, stock))
check("scope pins exactly (hipHostRegister)", t.is_pinned())
check("scope restores PLEVocabParallelEmbedding", ple_layer.PLEVocabParallelEmbedding is stock)
check("scope restores Tensor.pin_memory", torch.Tensor.pin_memory is pin)
os.environ["R9K_PLE_FORMAT"] = "stock"
with Q.construction_scope():
    check("scope leaves stock PLE for non-int6 checkpoints", ple_layer.PLEVocabParallelEmbedding is stock)

# --- the one remaining monkeypatch: target still exists and is ours ------------------------------------------
from vllm.v1.spec_decode import llm_base_proposer as lbp
src = open(lbp.__file__).read()
check("MTP allowlist target: SpecDecodeBaseProposer.allowed_attn_types still exists", "allowed_attn_types" in src)
check("MTP allowlist patch installed", getattr(lbp.SpecDecodeBaseProposer.__init__, "__r9k_patch__", None) == "mtp_allowlist")
check("MTP allowlist still needed (else retire the patch)", "QSAForwardMetadata" not in src)

# --- kernels plugged into stock objects -----------------------------------------------------------------------
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxfp4 import (
    CompressedTensorsW4A4Mxfp4)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4MoEMethod)
from r9700_vllm.linear.mxfp4 import R9700Mxfp4LinearKernel
check("stock dense MXFP4 scheme exposes .kernel", "kernel" in CompressedTensorsW4A4Mxfp4.__init__.__code__.co_names)
check("R9kMxfp4MoEMethod subclasses stock MoE method", issubclass(R9kMxfp4MoEMethod, CompressedTensorsW4A4Mxfp4MoEMethod))
check("stock MoE pwal still builds via make_mxfp4_moe_kernel",
      "make_mxfp4_moe_kernel" in open(sys.modules[CompressedTensorsW4A4Mxfp4MoEMethod.__module__].__file__).read())

print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
