import os
import transformers
from transformers import AutoModelForCausalLM

LF_MODEL_RANDOM_INIT = int(os.getenv('LF_MODEL_RANDOM_INIT', '0'))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', '0'))

@classmethod
def from_pretrained_for_throughput(self, *args, **kwargs):

    config = kwargs.pop("config", None)
    if config is not None:
        model = AutoModelForCausalLM.from_config(config)
    else:
        kwargs['state_dict'] = {}
        kwargs['low_cpu_mem_usage'] = False
        kwargs['device_map'] = None
        model = self.from_pretrained_for_throughput_origin(*args, **kwargs)

    return model

def patchAutoModelForCausalLM():

    if LF_MODEL_RANDOM_INIT == 1:
        transformers.AutoModelForCausalLM.from_pretrained_for_throughput_origin = transformers.AutoModelForCausalLM.from_pretrained
        transformers.AutoModelForCausalLM.from_pretrained = from_pretrained_for_throughput

        if LOCAL_RANK == 0:
            print(f"TILEARN - LLAMA FACTORY - LF_MODEL_RANDOM_INIT:{LF_MODEL_RANDOM_INIT}, patchAutoModelForCausalLM done!!!")

