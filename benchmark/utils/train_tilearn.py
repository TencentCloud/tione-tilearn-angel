try:
    from llmtuner.train.tuner import run_exp
except:
    from llamafactory.train.tuner import run_exp

import tilearn.llm.hybrid_parallel
from tilearn.llm.transformers.models import patch_models
#from tilearn.llm.memory.cpu.memory_optimize import memory_optimize
from patch_llama_factory import patchAutoModelForCausalLM


def main():

    patchAutoModelForCausalLM()
    patch_models()
    #memory_optimize()

    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
