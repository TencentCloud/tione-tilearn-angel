## Tilearn Llama3 训练加速3D并行案例集

### 1. 环境准备
#### 1.1 运行镜像
推荐使用平台内置镜像：tilearn-llm1.0-torch2.1-angel-vllm1.0-py3.10-cuda12.1-gpu, 在自定义镜像种使用该功能请参考附录

镜像中已经安装相关加速包tilearn-llm、tilearn.ops 版本如下
```bash
tilearn-llm>=0.9.3
tilearn.ops>=0.2.1.172
```

#### 1.2 下载代码
```bash
apt install git
git clone https://github.com/TencentCloud/tione-tilearn-angel.git
```
github网络不稳定时，可直接从cos下载
```bash
wget https://tione-public-cos-1308945662.cos.ap-shanghai.myqcloud.com/tilearn/hybrid_parallel/tione-tilearn-angel.tar.gz
tar -zxvf tione-tilearn-angel.tar.gz
```

#### 1.3 准备数据集和模型配置文件
```bash
cd tione-tilearn-angel/benchmark && bash prepare.sh
```
当前demo采用真实数据+模型参数随机初始化进行测试，仅提供模型config。若要读取huggingface模型参数，请参考附录进行修改

### 2. 开始性能测试
测试demo基于llamafactory构建，测试baseline性能数据
```bash
# tione-tilearn-angel/benchmark/llama3/8B
cd llama3/8B
### A800 80G GPU
bash single_node.sh
### A100 40G GPU
bash single_node_40g.sh
```
测试tilearn性能数据
```bash
# tione-tilearn-angel/benchmark/llama3/8B
### A800 80G GPU
bash single_node_tilearn.sh
### A100 40G GPU
bash single_node_tilearn_40g.sh
```

llama3 8b 8xA800 80G测试结果如下

Method | mbs | grad acc | gloabal bs | seqlength | GPU Mem | sec/iter | tokens/sec/gpu 
---- |-----|-----|------|-----------| ---- | ---- | ---- 
baseline flashattn2 | 1 | 1 | 8    | 4096      | 65805MiB | 1.3 | 3150.8
baseline flashattn2 | 1  | 16 | 128 |  4096     | 67915MiB | 22.03 | 2974.9
tilearn.llm tp=1 pp=2 | 1 | 32 | 128  | 4096      | 51585MiB | 16.23 | 4037.9

llama3 8b 8xA100 40G测试结果如下

Method | mbs | grad acc | gloabal bs | seqlength | GPU Mem | sec/iter | tokens/sec/gpu 
---- |-----|-----|-------------|-----------| ---- | ---- | ---- 
baseline flashattn2 | 1 | 1 | 8           | 4096      | 40081MiB | 1.67 | 2452.7
baseline flashattn2 | 1  | 16 | 128 |  4096     | 40127MiB | 30.79 | 2128.5
tilearn.llm tp=2 pp=2 | 1 | 64 | 128   | 4096      | 36343MiB | 19.65 | 3335.2


### 附录
详细配置请参考文档 [tilearn-llm](https://pypi.org/project/tilearn-llm/), [T平台训练加速功能介绍](https://cloud.tencent.com/document/product/851/76701)

#### 可选：自定义镜像使用tilearn-llm、tilearn.ops

镜像中的 torch.__version__=='2.1.2'，其他版本请联系加速团队
```bash
# tilearn-llm>=0.9.3
# tilearn.ops>=0.2.1.172
pip3 uninstall -y tilearn.llm tilearn.ops
pip3 install tilearn-llm==0.9.3 -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip3 install tilearn.ops==0.2.1.172 -i https://g-bnvx3728-pypi.pkg.coding.net/tione/tilearn/simple
wget https://tione-public-cos-1308945662.cos.ap-shanghai.myqcloud.com/tilearn/hybrid_parallel/colossalai-0.3.4.1-cp310-cp310-linux_x86_64.whl
pip3 install colossalai-0.3.4.1-cp310-cp310-linux_x86_64.whl
```

#### 可选：平台镜像内更新tilearn-llm、tilearn.ops版本
在镜像内更新最新的 tilearn.llm 和 tilearn.ops 包
```bash
# tilearn-llm>=0.9.3 
# tilearn.ops>=0.2.1.172
pip3 uninstall -y tilearn.llm tilearn.ops
pip3 install tilearn-llm==0.9.3 -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip3 install tilearn.ops==0.2.1.172 -i https://g-bnvx3728-pypi.pkg.coding.net/tione/tilearn/simple
```

#### 可选：关闭模式参数随机初始化
当前采用模型随机初始化测试，如果要关闭，在single_node.sh和single_node_tilearn.sh脚本内，设置LF_MODEL_RANDOM_INIT环境变量为0
此时MODEL_PATH路径下需要提供huggingface完整的模型config和模型参数
```bash
### Demo Args
# llama factory model random initialization
export LF_MODEL_RANDOM_INIT=0
MODEL_PATH=$BASE_PATH/models/$MODEL_NAME
```

#### 可选：训练代码配置tilearn 3d并行
方法一：环境变量配置
```bash
export TILEARN_HYBRID_TP_SIZE=1
export TILEARN_HYBRID_PP_SIZE=2
```
训练代码
```python
### 计算优化
from tilearn.llm.transformers import LlamaForCausalLM
from tilearn.llm.transformers import AutoModelForCausalLM
### 3D并行
import tilearn.llm.hybrid_parallel

def main():
    ### 模型接口与标准huggingface一致
    model = AutoModelForCausalLM.from_pretrained(...)
    
    run_exp()
```
方法二：python代码配置
训练代码
```python
### 计算优化
from tilearn.llm.transformers import LlamaForCausalLM
from tilearn.llm.transformers import AutoModelForCausalLM
### 3D并行
import tilearn.llm.hybrid_parallel
from tilearn.llm.hybrid_parallel.accelerate.config import config

def main():
    ### 在huggingface training_args初始化之前
    config.update(tp_size=1, pp_size=2)
    
    ### 模型接口与标准huggingface一致
    model = AutoModelForCausalLM.from_pretrained(...)
    
    run_exp()
```

#### 可选：修改demo训练代码

我们采用开源库[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/) 作为demo案例集，LLaMA-Factory对huggingface trainer进行封装，可在以下文件内修改训练代码
```python
# vim /opt/conda/lib/python3.10/site-packages/llmtuner/train/sft/workflow.py

def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    # ...
    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args),
    )
    # ...
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
```

### Acknowledgement
本案例集受益于 [ColossalAI](https://github.com/hpcaitech/ColossalAI), [transformers](https://github.com/huggingface/transformers), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/),
[flash-attention](https://github.com/Dao-AILab/flash-attention) 和 [pytorch](https://github.com/pytorch/pytorch), 感谢以上作者的付出。 
