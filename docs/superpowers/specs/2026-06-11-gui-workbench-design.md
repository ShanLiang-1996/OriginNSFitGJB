# OriginNSFitGJB GUI 工作台设计

日期：2026-06-11

## 背景

OriginNSFitGJB 当前是 Python 命令行工具，用于批量读取疲劳寿命与应变/应力响应数据，执行 GJB/Z 18A 9.3.2 简化应变寿命拟合，导出 CSV/JSON/Excel 审计结果，并可通过 Origin 自动化生成工程图和 Origin 项目。

现有代码重点包括：

- `src/originnsfitgjb/cli.py`：命令行参数、批处理流程、输出文件写入、Origin 调用。
- `src/originnsfitgjb/data_loader.py`：输入文件发现、表格读取、列自动识别。
- `src/originnsfitgjb/gjb.py`：GJB/Z 18A 拟合、审计步骤、决策日志。
- `src/originnsfitgjb/audit.py`：审计 CSV/JSON/Excel/清单输出。
- `src/originnsfitgjb/origin_client.py`：Origin 自动化、图表和项目文件生成、进程清理。

首版 GUI 的目标不是替换计算内核，而是给工程用户提供一个 Windows 桌面入口，并为后续功能扩展保留清晰边界。

## 决策

- 交付形态：Windows 桌面 exe。
- GUI 技术：PySide6/Qt。
- 首版使用场景：操作员一键全流程运行。
- Origin 策略：主按钮直接执行完整流程，包括 Origin 自动化。
- 扩展方向：插件式工作台底座，首版只注册 GJB/Z 18A 分析模块。
- 配置保存：保存最近任务和参数预设。
- 界面语言：中文为主。

## 总体架构

项目拆成三层。

第一层是核心计算层。保留现有 `data_loader`、`gjb`、`audit`、`origin_client` 的职责，继续负责读表、拟合、审计输出、Origin 项目和图表生成。

第二层是应用服务层。新增不依赖 GUI 的服务模块，例如 `src/originnsfitgjb/analysis_service.py`。服务层接收结构化配置，执行现有主流程，并返回结构化结果、输出文件列表、阶段状态和错误信息。CLI 和 GUI 都调用这一层。

第三层是 GUI 工作台层。新增 `src/originnsfitgjb/gui/` 包，用 PySide6 实现主窗口、模块注册表、GJB 分析页面、运行进度、日志面板、最近任务和参数预设。

目标边界：

- CLI 继续可用，并且与 GUI 共用同一服务层。
- GUI 不直接拼接命令行字符串。
- Origin 自动化仍封装在 `OriginClient` 内部。
- 后续新增功能时优先新增模块和服务，不重写主窗口骨架。

## 首版 UI

打开程序后直接进入可操作工作台，不做营销式首页。

左侧为窄导航栏。首版显示 `GJB/Z 18A 分析`，后续可加入 `数据处理`、`报告/审计`、`设置`。未实现模块不应制造强烈空壳感，可以先隐藏或以低调禁用状态出现。

主区域包含三组内容。

第一组是输入与列设置：

- 输入路径，可选择目录或后续扩展为单文件。
- 文件模式，例如 `*.csv`、`*.xlsx`，支持多个模式的扩展。
- 输出目录。
- 寿命列、响应列、状态列、分组列。
- 首版列名允许手填；后续可通过预览样例文件提供下拉选择。

第二组是分析参数与 Origin 选项：

- 置信度。
- 拟合点数。
- 异常值模式：自动剔除或仅报告。
- 是否写审计输出。
- 是否写审计 workbook。
- 是否隐藏 Origin。
- 是否生成线性化图。
- 是否显示 runout 箭头。
- Origin 项目路径。
- Origin 图模板路径。

第三组是运行与结果：

- 主按钮：`开始全流程分析`。
- 阶段进度：发现文件、拟合、写 CSV/审计、生成 Origin 项目、完成或失败。
- 日志面板：显示运行摘要和技术日志。
- 完成后展示常用输出入口，例如 `gjb_summary.csv`、审计目录、Origin 项目、图片目录、`origin_automation.log`。

## 运行数据流

GUI 点击主按钮后构造 `AnalysisConfig`，包含路径、文件模式、列设置、GJB 参数、审计选项和 Origin 选项。

GUI 调用：

```text
analysis_service.run_analysis(config, progress_callback, log_callback)
```

服务层按阶段执行：

1. 创建输出目录。
2. 发现输入文件。
3. 读取表格并解析列。
4. 对每个表执行 GJB/Z 18A 拟合。
5. 写 CSV 输出。
6. 按配置写审计输出。
7. 按配置生成 Origin 项目和图片。
8. 返回结构化结果。

CLI 改为把 argparse 参数转换为同一个 `AnalysisConfig`，然后调用服务层。这样 GUI 抽象不会导致 CLI 和 GUI 两套流程分叉。

## 错误处理

错误分三类处理。

配置错误包括路径不存在、参数非法、列名找不到等。GUI 应以中文提示用户修改输入，并避免继续执行明显无效的流程。

数据或拟合错误可能只影响某个表。服务层应记录失败表和原因，其他表能继续时继续执行。结果页展示成功数量、失败数量和失败明细。

Origin 自动化错误不应掩盖 Python 侧输出。因为首版主按钮执行完整流程，GUI 需要明确显示：Python 输出已完成，但 Origin 生成失败或部分失败。界面提供 `origin_automation.log` 入口。

日志保留完整技术细节；主界面给用户中文摘要。

## 扩展机制

新增 `gui/modules`，每个模块提供：

- 模块 ID。
- 中文名称。
- 页面组件工厂。
- 默认配置。
- 运行服务入口。

首版模块 ID 为 `gjb18a`，中文名称为 `GJB/Z 18A 分析`。

后续可按同一方式加入：

- 数据清洗模块。
- 列映射模板模块。
- 报告/审计模块。
- 其它标准或拟合模型模块。

主窗口只依赖模块注册表，不直接知道每个模块的业务细节。

## 配置保存

使用本地 JSON 保存用户配置，建议路径：

```text
%APPDATA%\OriginNSFitGJB\settings.json
```

保存内容：

- 最近输入路径。
- 最近输出路径。
- 最近文件模式。
- 最近列名设置。
- 常用分析参数。
- Origin 选项。
- 审计选项。
- 窗口尺寸。

用户本地配置不提交到仓库。

## 测试策略

优先测试服务层，确保 GUI 引入后现有 CLI 输出不变。

需要覆盖：

- `AnalysisConfig` 到现有流程参数的转换。
- CLI 调用服务层后的输出兼容性。
- dry-run 与完整流程的阶段事件。
- 配置保存和读取。
- 模块注册表。
- Origin 自动化失败时，Python 输出保留且错误信息可见。

现有测试 `tests/test_gjb_audit.py` 和 `tests/test_origin_cleanup.py` 必须继续通过。

GUI 本体首版可以以手工验证为主，但服务层和配置层需要自动化测试。

## 非目标

首版不实现复杂项目管理。

首版不做本地 Web 服务或浏览器内核 UI。

首版不新增新的 GJB 标准、拟合算法或报告模板。

首版不强制用户先 dry-run 再生成 Origin；主按钮直接执行完整流程。

## 实现默认值

后续实现按以下默认值展开：

- 新增 `requirements-gui.txt`，放置 PySide6 及 GUI 专用依赖；`requirements.txt` 继续服务 CLI 和核心计算。
- `requirements-build.txt` 引用 GUI 依赖，保证打包环境能构建桌面 exe。
- 离线部署同步补齐 PySide6 相关 wheel，避免工程机联网安装。
- 首版输入模型保持与现有 CLI 一致：选择输入目录和文件模式；单文件选择作为后续增强。
- 打包产物保留当前控制台 exe，同时新增 GUI exe。这样现有命令行自动化和脚本用法不受影响。
