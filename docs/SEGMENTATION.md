# 分割模型与接入状态

## 2026-09-18 更新

官方权重已下载并在 RTX 4090 上完成 3 个 CAMUS A4C 病例、6 张图像验证。
SHA256：`b32ea45b8200e11298ff4021166ff8d187204fb0c27a937bcdb4c518ffcbe1ed`。
mean=`[31.834011,31.95879,32.082172]`，std=`[48.866325,49.137333,49.361984]`，
来源为官方 EchoNet-Strain `Segmentation Analysis/Vid_to_Strain.py` 第 615 行的默认配置：
https://github.com/echonet/strain/blob/main/Segmentation%20Analysis/Vid_to_Strain.py#L615 。

模型固定为一类 DeepLabV3-ResNet50，输入 PIL 双线性 resize 到 112x112，
0-255 RGB 标准化；输出 logits 双线性恢复原始尺寸后以 0 为阈值。
重采样实现属于本项目适配策略，跨域 CAMUS 结果不代表原 EchoNet 测试集性能。
`weights_only=True` 成功读取 checkpoint，严格加载 `state_dict`，去除 `module.` 前缀。
没有利用本次 CAMUS 标签调参。平均 Dice=0.8509、IoU=0.7440，FAC MAE=8.14 个百分点。
推荐运行配置为 `configs/echonet.official.json`；pending 文件只保留作历史空模板。

以下为初始阶段记录，其未下载/未验证状态已被本节更新取代。

本阶段选择 EchoNet-Dynamic 官方 DeepLabV3-ResNet50 左室腔分割模型。

2026-09-09 服务器拉取的上游 commit：`e01d02947b20604b0ba1a427f3cf6f4890dc4e9c`。
该 commit 的源码许可为 MIT；权重和数据的使用条款仍需分别确认。

- 官方仓库：https://github.com/echonet/dynamic
- 权重发布：https://github.com/echonet/dynamic/releases/tag/v1.0.0
- 候选权重：https://github.com/echonet/dynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt
- 官方构建代码：https://github.com/echonet/dynamic/blob/master/echonet/utils/segmentation.py
- 上游许可：https://github.com/echonet/dynamic/blob/master/LICENSE

选择理由：目标与 LV cavity 一致，有公开代码和预训练权重入口，同一网络可处理 ED/ES。
上游模型主要针对 A4C；其他切面不能直接声称可用。LV cavity 也不等于 myocardium。

本次只拉取源码，建立 lazy adapter，不下载大权重、不安装 Torch、不在 2 GB 服务器上加载网络。
后续验证必须核对：实际权重可下载性及 SHA256、许可、训练 RGB mean/std、112x112
预处理、输出阈值、DataParallel 前缀以及在真实 ED/ES 图像上的轮廓。
上游训练脚本由训练集计算 mean/std，因此本项目要求显式配置，不套用 ImageNet 参数。

本轮无需 GPU 的主路径为预计算双 mask。mask 必须声明来源和前景标签；空 mask 直接失败。
真实图像及标注尚未确定，因此合成图像只验证数据流和算术，不报告临床分割性能。

`configs/echonet.pending.json` 是未配置模板，mean/std 故意为空；必须核对后填写才能运行。
真实双 mask 输入可参考 `configs/study.example.json`。
