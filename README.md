- ragq目录是依赖 https://github.com/yynil/RaqQ 将其进行了改造并增加了一些新的功能
- third_party_packages/目录是一些其它服务包
   - rwkv_peft 是微调包 https://github.com/JL-er/RWKV-PEFT
   - rwkv_lm_ext https://github.com/yynil/RWKV_LM_EXT/tree/main

#### 模型下载
请从https://huggingface.co/BlinkDL下载需要的rwkv模型，然后在配置文件中修改本地地址

#### 服务启动

配置文件为 ragq.yml
```shell
cd ragq
python3 service.py

```

#### 客户端测试
```shell
cd ragq
python3 run_test.py test_cache_client test_file_client
表示执行测试test/test_cache_client.py和 test/test_file_client.py的用例

```
