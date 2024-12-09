#pip3 install -r requirements.txt

rm models.tar.gz data.tar.gz
wget https://tione-public-cos-1308945662.cos.ap-shanghai.myqcloud.com/tilearn/hybrid_parallel/models.tar.gz
wget https://tione-public-cos-1308945662.cos.ap-shanghai.myqcloud.com/tilearn/hybrid_parallel/data.tar.gz
tar -zxvf models.tar.gz
tar -zxvf data.tar.gz
