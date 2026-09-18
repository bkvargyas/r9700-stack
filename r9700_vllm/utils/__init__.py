
_SHAPES: set = set()


def note_shape(kind: str, N: int, K: int) -> None:
    """Log each distinct (kernel path, N, K) once per process at load -- the tuning inventory (tuning/)."""
    key = (kind, int(N), int(K))
    if key in _SHAPES:
        return
    _SHAPES.add(key)
    from vllm.logger import init_logger
    init_logger("vllm.r9700_vllm.shapes").info("r9700 shape: %s N=%d K=%d", kind, N, K)
