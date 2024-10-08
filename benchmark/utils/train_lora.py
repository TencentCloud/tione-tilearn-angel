try:
    from llmtuner.train.tuner import run_exp
except:
    from llamafactory.train.tuner import run_exp

from patch_llama_factory import patchAutoModelForCausalLM


def main():

    patchAutoModelForCausalLM()

    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
