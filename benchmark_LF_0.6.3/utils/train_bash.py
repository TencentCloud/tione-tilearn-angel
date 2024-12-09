from llmtuner import run_exp

from patch_llama_factory import patchAutoModelForCausalLM

def main():

    patchAutoModelForCausalLM()

    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
