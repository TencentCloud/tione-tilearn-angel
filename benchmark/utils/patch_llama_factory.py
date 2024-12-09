import os
import transformers
from transformers import AutoModelForCausalLM, TrainingArguments

LF_MODEL_RANDOM_INIT = int(os.getenv('LF_MODEL_RANDOM_INIT', '0'))
LF_UT_TEST = int(os.getenv('LF_UT_TEST', '0'))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', '0'))

@classmethod
def from_pretrained_for_throughput(self, *args, **kwargs):

    config = kwargs.pop("config", None)
    if config is not None:
        attn_implementation = getattr(config, "_attn_implementation", None)
      
        #from colossalai.lazy import LazyInitContext
        #from colossalai.utils import get_current_device
        #init_context = LazyInitContext(default_device=get_current_device())
        #with init_context:
        model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_implementation, trust_remote_code=True)
    else:
        kwargs['state_dict'] = {}
        kwargs['low_cpu_mem_usage'] = False
        kwargs['device_map'] = None
        model = self.from_pretrained_for_throughput_origin(*args, **kwargs)

    return model

def save_model_for_throughput(self, *args, **kwargs):
    print(f"TILEARN - LLAMA FACTORY - LF_UT_TEST:{LF_UT_TEST}, Skip save_model for UT!!!")
    return 

def _save_checkpoint_for_throughput(self, *args, **kwargs):
    print(f"TILEARN - LLAMA FACTORY - LF_UT_TEST:{LF_UT_TEST}, Skip _save_checkpoint for UT!!!")
    return

def patchAutoModelForCausalLM():

    if LF_MODEL_RANDOM_INIT == 1:
        transformers.AutoModelForCausalLM.from_pretrained_for_throughput_origin = transformers.AutoModelForCausalLM.from_pretrained
        transformers.AutoModelForCausalLM.from_pretrained = from_pretrained_for_throughput
        if LOCAL_RANK == 0:
            print(f"TILEARN - LLAMA FACTORY - LF_MODEL_RANDOM_INIT:{LF_MODEL_RANDOM_INIT}, patchAutoModelForCausalLM done!!!")

    if LF_UT_TEST == 1:
        from tilearn.llm.llamafactory.patch_parse_train_args import patchParseTrainArgs 
        from tilearn.llm.llamafactory.patch_workflow import patchWorkflow
        patchParseTrainArgs()
        patchWorkflow()

        transformers.trainer.Trainer.save_model_ut_origin = transformers.trainer.Trainer.save_model
        transformers.trainer.Trainer.save_model = save_model_for_throughput
        transformers.trainer.Trainer._save_checkpoint_ut_origin = transformers.trainer.Trainer._save_checkpoint
        transformers.trainer.Trainer._save_checkpoint = _save_checkpoint_for_throughput

        if LOCAL_RANK == 0:
            print(f"TILEARN - LLAMA FACTORY - patchParseTrainArgs && patchWorkflow && save_model done!!!")



