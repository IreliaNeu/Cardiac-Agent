# 三角色单轮心超分析：设计与运行

## 本阶段目标

输入一位患者的 CAMUS A4C 舒张末期 ED 和收缩末期 ES 图像，输出原图、模型预测的
左室心腔分割图、叠加图和 Qwen 生成的研究报告。每病例仅允许 Planner、Analyst、Reviewer
各发言一次，不追加讨论轮次。三角色目前共用同一 Qwen API 模型，但提示、输入和输出契约不同。

本阶段是有界工具规划和报告审查的工程实现，不是自由协商的多智能体系统，也不是完成临床验证的诊断系统。
原有确定性流水线保留，可作为后续消融实验的基线。

## 输入、交互与输出

```text
ED/ES 原图 + 视图/时相信息
    |
    +--> Planner：读取非识别性元数据，选择 area_only 或 area_and_motion
    |        输出 motion=none/farneback、focus、简短选择理由
    |
    +--> 工具执行：EchoNet 分割 -> 面积/FAC -> 可选稀疏光流 -> QC
             |
             +--> 结构化证据：精确数值、单位、RWMA 可用状态、适用边界
                         |
                 Analyst：读取计划与证据，生成报告草稿
                         |
                 Reviewer：读取计划、证据和草稿，单次审查并输出最终报告
                         |
                 契约/数值检查 + 硬性 QC 覆盖 -> report.md / report.json

预测 mask + 人工 mask -> 独立评估 Dice、IoU、FAC 误差
                        评估结果不返回角色提示
```

Planner 的选择实际控制 `pipeline.run(..., deformation=...)`，不是仅生成一段“计划文字”。
单轮中 Reviewer 可以修改报告和建议人工复核，但不能改动测量值、重新分割或再次询问 Analyst。
硬性 QC 出现心腔面积不下降或分割触边时，程序将最终状态强制设为 `refer`；其他报告也只能标记
`limited` 或 `refer`，没有“临床确诊”状态。

## 模块职责

| 文件 | 职责 |
|---|---|
| `src/cardiac_agent/discussion.py` | 三角色契约、一次交互、请求缓存、证据校验、原图/分割图导出、最终报告 |
| `src/cardiac_agent/explanation.py` | 共享 Qwen HTTP 客户端；保留原有受约束证据排序接口 |
| `src/cardiac_agent/pipeline.py` | 确定性分割、测量、RWMA 接口、Knowledge Bridge 与基础审计 |
| `src/cardiac_agent/roi.py` | 官方 EchoNet DeepLabV3-ResNet50 预测左室心腔 |
| `src/cardiac_agent/function.py` | 像素面积、FAC、面积比、可选光流统计和 QC |
| `scripts/three_role_batch.py` | 官方数据下载、固定病例选择、并发批处理、独立评估、索引与汇总 |
| `scripts/audit_three_role_batch.py` | 只读验收：哈希、图像、三次调用、数值、标注隔离、数量与失败分母 |
| `tests/test_discussion.py` | 单轮限制、断点缓存、参考隔离、QC 覆盖及数值校验测试 |

工具阶段内部的 `execution/.../explanation.json` 是原有确定性基线说明。
交付的模型生成报告是病例目录根部 `report.md`，对应 Reviewer 输出，二者不要混淆。

## 数据和隐私边界

- 100 条指 100 位不同患者，每位一对 ED/ES 图像，共 200 张原图，不是 100 张单图。
- 按官方患者文件夹名称升序选择前 100 例，仅使用 A4C；不按质量或模型得分筛选、不静默替换失败病例。
- 原始 NIfTI、人工标注、配置和下载 URL/SHA256 留在服务器数据目录；PNG 是导入器生成的显示图。
- 人工 mask 用于导入时网格/标签有效性检查，以及讨论结束后的独立评估；不作为预测输入，不进入 Agent 证据。
- 发送给 API 的内容只有非识别性视图/时相元数据、预测产生的医学指标、QC 和角色文本。
- 不发送患者 ID、图像路径、原图、mask 或人工评估分数。API 密钥不进入日志、仓库和归档。
- 原图、实验结果、权重、模型响应只留在服务器。GitHub 仅保存代码、配置、测试和文档中的汇总数字。

## 报告可信度边界

每个 Analyst/Reviewer 响应必须包含全部原始 evidence 键值，不允许新增或修改。
正文中的阿拉伯数字必须对应某个证据值，允许按正文显示精度进行十进制 half-up 舍入。
报告末尾追加不可由模型删除的精确证据表、QC 和固定适用边界。

这是词法级校验，不是医学事实验证：不能保证某个正确数字与正确指标关联，不能排除中文数字表达的错误，
不能判断单位误用或诊断语义错误。模型 Reviewer 也不构成独立临床真值。需要医工老师后续盲评报告。

FAC 是双帧二维心腔面积变化，不是 LVEF；两帧稀疏心腔光流不是心肌应变；目前没有经验证的 RWMA 分类器。
CAMUS 人工 mask 可验证分割，但不能替代 RWMA 标注。EchoNet 到 CAMUS 属于跨数据集应用，不能将流程成功率
解释为临床准确率。即使某例分割 Dice 很低，当前 Agent 也看不到人工 Dice，仍可能输出 limited；这是待优化的问题。

## 运行方案

在项目根目录执行，先按 README 建立 GPU Conda 环境并放置官方权重：

```bash
conda activate cardiac-agent-gpu
python -m pytest -q
ruff check src scripts tests
python scripts/three_role_batch.py \
  --data data/camus-100-20260927 \
  --output runs/three-role-100-20260927 \
  --count 100 --workers 4 \
  --env /root/autodl-tmp/RS-Agent/.env
python scripts/audit_three_role_batch.py runs/three-role-100-20260927 --count 100
```

API 默认 SiliconFlow `Qwen/Qwen3-30B-A3B-Instruct-2507`，可通过环境变量切换。
四个线程并发下载和调用 API，GPU 推理使用共享模型及锁串行执行，避免模型重复加载。
不要在批次运行中修改 `src/cardiac_agent`、配置、提示或输入。

同一输出目录支持断点恢复：已完成病例先校验全部文件；已收到的角色响应按请求哈希复用；
已完成工具阶段按输入、代码、模型身份及校验和复用。对已启动但结果未落盘的 API 调用不自动重放，
以避免断网后重复语义讨论。传输错误、限流和部分服务错误最多尝试三次，次数会记录，可能重复提供方请求。
因此“一轮”是三个角色回合，不是保证网络层恰好只有三次 HTTP 请求。

失败病例保留目录及失败类型，汇总记录总请求数、成功数和失败数，并使进程返回非零退出码。
需要改提示或换模型时使用新的输出目录；不要删除失败痕迹伪造一次成功的执行。

## 交付结构

```text
runs/three-role-100-20260927/
  selection.json              固定患者选择及官方来源
  download_status.json        下载成功与失败情况
  index.csv                   病例、报告路径、状态及独立评估值
  summary.json                批次汇总，含失败分母
  results.json                病例级结果
  README.md                   结果阅读说明
  checksums.json              整个目录的 SHA256
  patientXXXX/
    original/ED.png ES.png
    segmentation/ED_mask.png ES_mask.png ED_overlay.png ES_overlay.png
    report.md report.json
    planner.json analyst.json reviewer.json agent_trace.json
    independent_evaluation.json
    execution.json execution/<uuid>/...
    result.json checksums.json
```

验收报告和批处理日志放在该目录外，避免修改已经封存的目录校验和。
原始 NIfTI 数据目录与结果目录需一起保留，内部审计记录含绝对路径。

## 后续优化与需要医工老师确认的事项

1. 首先建立不依赖人工 mask 的分割质量控制：多连通域、形状异常、解剖合理性及模型不确定性。
2. 确定正式任务是分割辅助、整体收缩指标还是 RWMA；不能以 FAC 替代 RWMA 标签。
3. 确定是否能提供完整视频、时相核验和心肌分区标注，决定是否开展可靠运动分析。
4. 由医工老师评价报告事实一致性、单位、适用边界和不当诊断；建立 blinded rubric。
5. 比较固定 pipeline、单 Agent、三角色单轮；加入相同 API/Token 预算对照，验证 Reviewer 的实际贡献。
6. 只有具备可靠工具/质量信号后，再引入有预算上限的重分割、主动补充检查及经审核的病例记忆。
7. 当前相同模型多角色存在相关错误，不应把一致意见当作独立证据。长期记忆和开放式自由讨论尚未实现。
