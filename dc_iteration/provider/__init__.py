from dc_iteration.provider.base import DecoderBase


def make_model(
    model: str,
    backend: str,
    dataset: str,
    batch_size: int = 1,
    temperature: float = 0.0,
    force_base_prompt: bool = False,
    instruction_prefix=None,
    response_prefix=None,
    dtype="bfloat16",
    trust_remote_code=False,
    tp=1,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    base_url=None,
    attn_implementation="eager",
    device_map=None,
) -> DecoderBase:
    if backend == "openai":
        from dc_iteration.provider.openai import OpenAIChatDecoder

        assert not force_base_prompt, f"{backend} backend does not serve base model"
        return OpenAIChatDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            base_url=base_url,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}. Only 'openai' is supported in dc_iteration.")
