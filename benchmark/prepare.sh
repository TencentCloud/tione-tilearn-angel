#pip3 install llmtuner jieba nltk rouge-chinese

pip3 install tilearn-llm==0.9.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install tilearn.ops==0.2.1.172 -i https://g-bnvx3728-pypi.pkg.coding.net/tione/tilearn/simple
pip3 install colossalai==0.3.6
pip3 install llmtuner==0.6.3

rm models.tar.gz data.tar.gz
wget https://tione-public-cos-1308945662.cos.ap-shanghai.myqcloud.com/tilearn/hybrid_parallel/models.tar.gz
wget https://tione-public-cos-1308945662.cos.ap-shanghai.myqcloud.com/tilearn/hybrid_parallel/data.tar.gz
tar -zxvf models.tar.gz
tar -zxvf data.tar.gz
