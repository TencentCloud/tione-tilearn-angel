## Benchmark UT 

```bash
### 所有模型测试混合并行+计算优化
bash run_dump_log.sh run_all_test.sh

### 所有模型测试deepspeed-zero30offload vs tilearn-autozero
bash run_dump_log.sh run_all_test_autozero.sh 

### 单个模型遍历各种配置
bash run_dump_log.sh run_one_test.sh 
```
