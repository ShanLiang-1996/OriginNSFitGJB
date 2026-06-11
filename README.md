# OriginNSFitGJB

OriginNSFitGJB 用于批量读取疲劳寿命与应变/应力响应数据，按 GJB/Z 18A 9.3.2 的简化应变寿命方法完成拟合、复核表导出，并自动操控 Origin 生成工程图与项目文件。

本项目只保留 GJB/Z 18A 9.3.2 Formula 136 相关流程。拟合关系为：

```text
log10(Nf) = A1 + A2 * log10(response - A4)
```

其中 `response` 使用输入数据中的应变或应力响应列；当前简化实现不拟合 A3，直接把该响应列作为等效应变输入。

## 环境准备

联网环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
```

离线环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\offline\install_offline.ps1
```

离线包说明见 [offline/README.md](offline/README.md)。`originpro`/`originpy` 需要目标 Windows 电脑已安装并激活 Origin 或 OriginPro。

## 数据格式

CSV、TSV、TXT、XLS、XLSX 均可作为输入。最小列要求：

- 寿命列：如 `life`、`N`、`寿命`、`cycles`
- 响应列：如 `strain`、`stress`、`应变幅`、`最大应变`、`应力幅`
- 状态列可选：如 `status`，值中包含 `runout`、`suspended`、`未失效` 等会按删失/停试点处理

示例：

```csv
specimen_id,strain,life,status
GJB-001,0.0120,520,failure
GJB-002,0.0105,780,failure
GJB-005,0.0065,3500,runout
```

## 只计算并导出 CSV

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb --input examples --output output --pattern gjb18a_strain_example.csv --status status --dry-run
```

如果列名无法自动识别，可显式指定：

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb --input data --output output --life "life" --response "strain" --status "status" --dry-run
```

## 审计输出

需要人工复核每一步计算时，启用审计模式：

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb --input examples --output output --pattern gjb18a_strain_example.csv --status status --dry-run --audit --audit-workbook
```

默认审计目录：

```text
output\audit\
output\audit\tables\<label>\
output\audit\json\<label>\
output\audit\gjb_decision_log.csv
output\audit\gjb_audit_workbook.xlsx
output\audit\gjb_manual_checklist.md
```

`output\audit\gjb_audit_workbook.xlsx` 面向不写程序的复核人员：每个 Step 工作表顶部写明本步骤目的、公式、输入列、输出列和判定规则，下方是逐行数据。`ManualCheck` Sheet 与 `gjb_manual_checklist.md` 列出可在 Excel 中手工复算的检查项。

`gjb_decision_log.csv` 记录流程判定，例如是否加权、A4 置信区间是否触发固定为 0、A2 是否仅备注为寿命-应变关系不显著、固定 A4 线性修正是否执行、异常值是否删除、runout 是否进入 MLE 的 `logsf` 项。

异常值模式：

```powershell
--outlier-mode auto         # 默认，保持自动剔除并迭代
--outlier-mode report-only  # 只报告候选异常值，不删除数据
```

即使 Origin 自动化失败，Python 侧 CSV、JSON、Excel 审计输出也会先完整保存。当前模型仍为：

```text
log10(Nf) = A1 + A2 * log10(response - A4)
```

其中 `response` 直接作为等效应变输入，当前项目不拟合 A3。

## 生成 Origin 项目

确认 dry-run 成功后，去掉 `--dry-run`：

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb --input data --output output --pattern "*.csv" --status status
```

默认输出：

```text
output\gjb_analysis.opj
output\figures\
```

可选参数：

```powershell
--project output\my_gjb_analysis.opju
--graph-template C:\path\to\gjb_template.otpu
--linearized-graph
--no-runout-arrows
--hidden-origin
```

## 输出文件

常用输出：

```text
output\gjb_summary.csv
output\gjb_fit_data.csv
output\gjb_runout_data.csv
output\gjb_curve.csv
output\gjb_level_stats.csv
output\gjb_initialols.csv
output\gjb_initialnls.csv
output\gjb_varianceanalysis.csv
output\gjb_refitdata.csv
output\gjb_refitresult.csv
output\gjb_parametersignificance.csv
output\gjb_fixeda4linearfit.csv
output\gjb_residuals.csv
output\gjb_outlieriterations.csv
output\gjb_finalmle.csv
output\gjb_likelihood.csv
output\gjb_modelchecks.csv
output\gjb_decisionlog.csv
output\gjb_finalresidualstatistics.csv
output\gjb_modelassessment.csv
output\gjb_r2documentstyle.csv
```

## 打包 exe

```powershell
.\.venv\Scripts\pyinstaller.exe OriginNSFitGJB.spec
```

打包结果：

```text
dist\OriginNSFitGJB.exe
```

## Origin 排错

如果 CSV 能生成但 Origin 项目失败：

1. 确认目标电脑已安装并能手动打开 Origin/OriginPro。
2. 确认没有首次启动弹窗、许可证弹窗或用户文件夹设置弹窗阻塞自动化。
3. 先运行 `--dry-run` 确认数据与拟合流程无误。
4. 查看 `output\origin_automation.log` 中的真实异常。
5. 程序退出时会先调用 Origin 正常退出；如果本次自动化新启动的 Origin 进程仍残留，会自动清理该进程。运行前已存在的 Origin 进程不会被清理。

## 目录结构

```text
src/originnsfitgjb/      GJB 拟合、数据读取、Origin 自动化源码
examples/                可直接运行的 GJB 示例数据
data/                    本地输入数据，默认不纳入 Git
output/                  运行输出，默认不纳入 Git
offline/                 离线部署脚本与 wheelhouse
OriginNSFitGJB.spec      PyInstaller 打包配置
```
