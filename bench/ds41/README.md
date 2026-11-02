# DeepSeek-V4.1 (EXL3) bring-up tools

Run against a serving `serve/ds41-exl3.sh` (URL `http://localhost:8080`, model `DeepSeek-V4.1`).

- `divergence.py URL MODEL [n]` -- greedy decode vs a fresh prefill of every prefix (needs prefix caching OFF): top-1
  agreement and the prefill's logprob of each decode token. Found the stock ROCm decode fallback bug (2026-11-02); a healthy
  server shows >= 22/24 agreement at |dlogprob| < 0.2.
- `needle.py URL MODEL ctx [depths]` -- planted facts in random filler, thinking off.
- `tprobe.py URL MODEL [streams] [max_tokens]` -- decode tok/s single and N streams (usage-based, spec-decode safe), warm prefill.
- `ask.py`, `ask2.py`, `ask3.py` -- correctness prompts (thinking on; sampling variants; long generations + mini needles).
- `gsm8k_ds41.py N CONC [MODEL]` -- GSM8K test split, greedy, thinking on, 6144-token budget.
