import os
import transformers
from transformers import AutoModelForCausalLM

LF_MODEL_RANDOM_INIT = int(os.getenv('LF_MODEL_RANDOM_INIT', '0'))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', '0'))

@classmethod
def from_pretrained_for_throughput(self, *args, **kwargs):
    ### set pretrained_model_name_or_path = None
    #if len(args) > 0:
    #    args_new = list(args)
    #    args_new[0] = None
    #    args = tuple(args_new)
    #if kwargs.get('pretrained_model_name_or_path', None) is not None: 
    #    kwargs['pretrained_model_name_or_path'] = None

    kwargs['state_dict'] = {}
    model = self.from_pretrained_for_throughput_origin(*args, **kwargs)

    return model

def patchAutoModelForCausalLM():

    if LF_MODEL_RANDOM_INIT == 1:
        transformers.AutoModelForCausalLM.from_pretrained_for_throughput_origin = transformers.AutoModelForCausalLM.from_pretrained
        transformers.AutoModelForCausalLM.from_pretrained = from_pretrained_for_throughput

        if LOCAL_RANK == 0:
            print(f"TILEARN - LLAMA FACTORY - LF_MODEL_RANDOM_INIT:{LF_MODEL_RANDOM_INIT}, patchAutoModelForCausalLM done!!!")

