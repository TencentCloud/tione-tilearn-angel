## Tione Tilearn Angel案例集

### 1. 训练加速

详细内容请参考文档 [tilearn-llm](https://pypi.org/project/tilearn-llm/), [T平台训练加速功能介绍](https://cloud.tencent.com/document/product/851/76701)

具体demo见benchmark文件夹和benchmark/README.md

### 附录
### 可选：自定义镜像使用tilearn-llm、tilearn.ops

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

### Acknowledgement
本案例集受益于 [ColossalAI](https://github.com/hpcaitech/ColossalAI), [transformers](https://github.com/huggingface/transformers), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/),
[flash-attention](https://github.com/Dao-AILab/flash-attention) 和 [pytorch](https://github.com/pytorch/pytorch), 感谢以上作者的付出。 
