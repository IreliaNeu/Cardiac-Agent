# Overview 对应架构与实现边界（2026-09-18）

独立目录 `/root/autodl-tmp/Cardiac-Agent`，独立 conda 环境 `cardiac-agent`。
另有独立 `cardiac-agent-gpu` 环境，RTX 4090 上已完成 EchoNet 分割验证。
完整技术说明、运行命令和结果见 `TECHNICAL_GUIDE.zh-CN.md`。

| Overview 组件 | 当前模块 | 实现与边界 |
| --- | --- | --- |
| ED/ES 输入 | schemas.py / cli.py | 同检查、同切面双帧，显式来源、spacing、mask 标签 |
| ROI-Agent | roi.py | 双 mask 校验；EchoNet 已在 3 个 CAMUS A4C 病例真实推理 |
| Function Agent | function.py | A_ED/A_ES、FAC、面积比，默认 intensity difference，光流可选 |
| RWMA Head | rwma.py | Predictor 协议与 unavailable/abstained 状态；真实权重待训练 |
| Knowledge Bridge | schemas.py / pipeline.py | 结构化 M→F→Y，artifact 路径/哈希、QC、模型来源 |
| Explanation Agent | explanation.py | Qwen API 组织事实，数值证据核验后确定性语言表达 |
| QA/Evaluation Agent | evaluation.py | 三个预设问题、算术一致性、可选独立 RWMA 标签评估 |
| 运行记录 | pipeline.py | 独立运行目录、原子 JSON、产物哈希、失败记录 |
| 对照评估 | benchmark.py | 人工/模型独立运行，分割及 FAC 误差，失败分母 |

## 数据流

1. 读取 Study 并检查图像、双 mask 与物理标定。人工 ED/ES 帧选择仍由输入提供。
2. ROI 获得 ED/ES cavity masks，Function 计算四个 Overview 特征字段。
3. FAC 与 area_ratio 数学共线，两者均保留便于对照文档；后续训练应去掉重复特征。
4. Overview 指定光流或特征差异。灰度 intensity difference 仅为早期工程基线，不能等同于学习特征差异。现已接入 Farneback 光流验证。
5. RWMA 未训练时明确 unavailable；不使用随机参数或 FAC 阈值冒充分类结果。
6. Knowledge Bridge 保存双 mask 引用、数值、预测状态、来源及限制。
7. Qwen 接收结构化事实，返回证据 ID 到值的映射或 claims 列表；2026-09-20 增加差异/光流方法、统计和单位契约，已在恢复的服务器上通过 4 次真实调用复验。
8. Explanation 验证条目和值后生成可复核文本。请求或内容失败会显式记录并回退模板。
9. QA 从结构化字段计算答案；缺失 RWMA 不算作阴性，也不计入准确率分母。

## 当前解释与评估能力

本阶段是严格事实模式，Qwen 的职责是组织证据顺序；开放式解释和用户自由提问尚未实现。
这样先打通 Overview 的消息契约。下一阶段可加入 claim-level 临床解释和医生 rubric，
但不能把病例级 RWMA、全局功能下降与 LVEF 混为一谈。
三项预设问题是工程占位版本，需与合作老师共同确定最终临床问题集。
QA 一致性只检验信息传递，不证明诊断正确。

## 后续计划

1. 已确定 CAMUS ED/ES 和人工 mask；下一步确认独立 RWMA 标签粒度与完整 cine 数据。
2. EchoNet 权重、normalization 和 A4C 推理已验证；需扩大样本并校验轮廓和跨域误差。
3. Dice/IoU、面积误差及批量对照已实现；补充 HD95 和正式患者级划分。
4. 按 Overview 训练 MLP RWMA Head；保留传统线性基线与图像特征融合消融。
5. 增加心肌/轮廓运动、完整 cine 对照、分类校准、区域特征及错误传播实验。
6. 增加 Qwen 自由问答、受约束临床说明与医生盲评；最后建设医学 Web 界面。

复用 RS-Agent 的分层与可追溯设计思想，不声称已移植其全部批处理、replay、多 Judge 工具。
