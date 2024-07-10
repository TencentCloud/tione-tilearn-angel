from llmtuner import run_exp

import tilearn.llm.hybrid_parallel
from tilearn.llm.transformers.models import patch_models
from patch_llama_factory import patchAutoModelForCausalLM

def main():

    patchAutoModelForCausalLM()
    patch_models()

    run_exp()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
