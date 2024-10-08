### Tione Tilearn Angel案例集-Qwen1 
#### 1. 计算训练加速
Qwen1 自动检测运行环境flash-attn，进行计算加速。请安装flash-attn，以开启计算加速能力

#### 2. 3D并行能力
目前Qwen1 支持数据并行、模型并行、流水线并行，详细内容请参考文档 [tilearn-llm](https://pypi.org/project/tilearn-llm/), 请使用0.9.10版本。[T平台训练加速功能介绍](https://cloud.tencent.com/document/product/851/76701) 
通过下面环境变量开启3D并行能力： 
方法一：环境变量配置
```bash
export TILEARN_HYBRID_TP_SIZE=1
export TILEARN_HYBRID_PP_SIZE=2
```
#### 3、关闭参数随机初始化模式（默认已开启）
Qwen1 默认开启参数随机初始化模式，即在single_node.sh和single_node_tilearn.sh脚本内，设置LF_MODEL_RANDOM_INIT环境变量默认为0
此时MODEL_PATH路径下需要提供huggingface完整的模型config和模型参数
```bash
### Demo Args
# llama factory model random initialization
export LF_MODEL_RANDOM_INIT=0
MODEL_PATH=$BASE_PATH/models/$MODEL_NAME
```
#### Acknowledgement 
本案例集受益于 ColossalAI, transformers, LLaMA-Factory, flash-attention 和 pytorch, 感谢以上作者的付出。