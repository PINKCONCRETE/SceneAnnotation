# RoboCoin Scene Annotator

基于开放词汇目标检测器和大语言模型的机器人场景标注自动化生成工具

# 安装

参考[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)

# 使用方法

1. 从NAS上下载：
浏览器打开：http://172.16.12.20:5000/
账号：BAAI_XSX
密码：BAAI123～
下载路径：/docker2/robocoin-datasets
其中每一个文件夹是一个数据集

2. 解压到/home/diy02/RoboCoin-scene-annotator/datasets，如
datasets
 |
 +-- realman_rmc_aidal_basket_storage_orange


3. 运行命令：
先打开终端
```bash

cd /home/diy02/RoboCoin-scene-annotator
conda activate dino

# repo_id这一行需要修改为要处理的数据集
python scripts/run_pipeline.py \
    --repo_id="agilex_cobot_decoupled_magic_take_out_the_bread" \
    --repo_root="/mnt/nas/synnas/docker2/robocoin-datasets" \
    --save_root="results/" \
    --camera="observation.images.cam_front_rgb" \
    --detector.type="grounding_dino" \
    --detector.device=cuda \
    --detector.visualize_first=5 \
    --language_model.type="ollama" \
    --language_model.model="deepseek-r1:14b" \
    --language_model.think=False                    
```


3.1. 生成物品的文本，在results/prompts/<repo_id>.txt中
3.2. 提取每个视频的第一帧，会打开一个窗口，需要从窗口中确认所有存在的物体，然后查看prompt文件中有没有缺失或多余的物体，如果有，则添加或删除。结果会放在results/frames/<repo_id>中
3.3. 目标检测，会展示前5个检测结果，观察有没有问题，如果有就修改prompt文件，如将black basket, yellow basket合为basket，通过这种方法减轻检测难度。结果会放在results/annotations/<repo_id>中
3.3.1. 如果检测有问题，按ctrl+c停止，修改文件，再重跑
3.4. 生成描述。结果会放在results/annotatinos_refinde/<repo_id>中。至此任务完成

特殊情况：
1. 如果有修改了也无法识别物体的情况，单独记录下来，跳过这个任务

# 致谢

感谢[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), [Ollama](https://github.com/ollama/ollama)等优秀的作品！