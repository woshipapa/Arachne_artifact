# VAST(video as storyboard from text)
* 支持T2V/I的基础模型训练
* 支持X2V的模型训练
## 拉取代码
请先添加本机ssh key到研发云: 代码仓库->设置我的ssh 公钥
确保/etc/resolv.conf 文件中添加了 nameserver 10.30.254.254
## 训练环境
使用如下docker进行或者任意python>=3.9的环境
### 物理机手动docker
```shell
sudo docker login harbor.telecom-ai.com.cn -u teleai -p Teleai123
sudo docker pull harbor.telecom-ai.com.cn/teleai-t2v/ncvr-torch2.5-cuda12.4:v0.4
```

挂载docker, 注意your_code_path 为此readme所在的文件夹路径
```shell
sudo docker run -it -v /path-to-your-code/vast:/workspace/vast -v /root:/root --ipc=host --privileged --gpus all --name vast 79c6a19c16b9 bash
```

```bash
# 安装本项目
pip install -v -e .

# 安装数据工具依赖, $USER是个人研发云的账号,即邮箱前缀
pip install git+ssh://${USER}@internal-git-host:29418/ANON/project/teleai_data_tool
```

## vast目录结构
```plaintext
|--configs
|  |--accelerate_configs                     -> accelerate trainer config
|  |--xxxx                                   -> 个人或任务config
|--projects                                  -> 各个任务临时工程文件夹，成熟后合并到vast中 
|  |--xxx                                    -> 模型实例
|  |  |--adaptors                            -> 适配trainer、transforms、loss等
|  |  |--configs                             -> 训练配置实例
|--tools                                     -> 训练工具启动入口
|  |--test.py                                -> vast测试脚本
|  |--train.py                               -> vast训练脚本
|--vast                                      -> S2V/VideoGen/VisionForge算法
|  |--datasets                                  -> vast下的数据集处理代码(原giga_datasets)
|  |--diffusion
|  |--models                                    -> 模型结构vae、textencoder、dit、loss、基础layers等(原giga_models)                        
|  |--pipelines                                 -> pipelines(原giga_models/pipelines)
|  |--train                                     -> vast下的训练代码(原giga_train/giga_train)
|--work_dirs                                    -> 实验结果，模型存储等默认文件夹
|--tests                                        -> 单元测试
|--requirements                                 -> 依赖管理
```

## training
```shell
python3 tools/train.py projects/hunyuanvideo/configs/hunyuanvideo_i2vhy.py
# show loss with tensorboard; 例如
tensorboard --logdir=work_dirs/hunyuanvideo_i2vhy/logs/hunyuanvideo_i2vhy
```
## inference 可视化
```shell
python3 projects/hunyuanvideo/infer_hunyuanvideo.py
```

## 多机训练

### 训练步骤

1. 拷贝代码至共享存储, 假设代码已拷贝至`/root/path-to-you/vast`

2. 修改vcjob.yaml中的参数配置
    - 修改meta.name作为实验名称

    -  修改worker replicas数, 即需要使用的8卡机数量

    -  分别修改master replicas和worker replicas的启动命令containers.args, 将`/root/path-to-you/vast`改写为你在共享存储的代码路径

3. 执行`sudo kubectl apply -f vcjob.yaml` 启动多机训练训练任务

PS: K8S控制节点为3/4/5, 所有k8s相关的任务需登录至IP主机位（尾号）为3/4/5的任意一台机器上执行。

### 其他常用命令

- `sudo kubectl get pods -o wide` 可查看当前在运行的训练任务信息

- `sudo kubectl logs [Pod-Name]` 可查看对应节点的log

- `sudo kubectl delete -f vcjob.yaml` 停止多机训练任务


## 开发规范

需要确认自己在https://internal-git-host/codehub/user/settings设置好ssh key, 方便安装ruff precommit库

我们采用[PEP8](https://www.python.org/dev/peps/pep-0008/)作为代码风格。

我们使用[pre-commit hook](https://pre-commit.com/)来：

`pre-commit`的配置存储在[.pre-commit-config](.pre-commit-config.yaml)中。

在clone代码仓库后，你需要安装并初始化 `pre-commit`：

```bash
pip install -U pre-commit
pip install ruff==0.8.4
```

并在仓库目录下运行：

```bash
pre-commit install
```

在顺利安装后，你每次提交代码时都会自动执行代码格式检查与自动格式化。

也可手动运行检查

```bash
pre-commit run --all-files
```
