# Code Execution Sandbox Plan

## 1. 背景

当前项目已经完成代码推理分支第一阶段：

```text
MBPP safe sample-only generation
→ sample inspection
→ static analysis
→ semantic analysis
→ README update
→ final_project_report update
```

当前稳定 tag：

```text
local-code-mbpp-safe-baseline-complete
```

当前阶段已经确认：

```text
1. lm-eval MBPP predict_only 会触发 Hugging Face code_eval 安全门；
2. 当前没有设置 HF_ALLOW_CODE_EVAL=1；
3. 当前没有执行模型生成代码；
4. 当前只做 safe sample-only generation；
5. 当前 MBPP limit=5 样本全部 executed=false；
6. 当前 static_clean = 5/5；
7. 当前 manual semantic likely correct = 2/5；
8. 当前 manual semantic suspicious/wrong = 3/5。
```

## 2. 为什么需要 sandbox

代码推理评估和数学推理评估不同。

数学推理评估一般是：

```text
模型生成答案
→ 抽取 final answer
→ 对比标准答案
```

代码推理评估一般是：

```text
模型生成 Python 代码
→ 执行模型生成代码
→ 运行测试用例
→ 计算 pass@1
```

因此代码推理评估会执行不可信代码。

模型生成代码可能包含：

```text
死循环
文件读写
网络请求
系统命令
高内存占用
高 CPU 占用
恶意 import
无限递归
删除文件
读取环境变量
```

所以不能直接在主机环境中无隔离执行模型生成代码。

## 3. 当前阶段不做的事情

当前阶段仍然不做以下事情：

```text
不设置 HF_ALLOW_CODE_EVAL=1
不直接运行 lm-eval code_eval
不直接执行 MBPP / HumanEval generated code
不直接计算正式 pass@1
不在主项目 Python 环境里执行模型生成代码
```

当前阶段只做方案设计。

## 4. 推荐执行路线

后续正式代码执行评估建议分三层推进。

### 4.1 第一层：继续 safe sample-only generation

先继续生成样本，不执行代码：

```text
MBPP safe sample-only
HumanEval safe sample-only
EvalPlus prompt/sample preparation
```

这一层的目标是确认：

```text
模型能否输出函数
函数名是否匹配
代码是否能被 ast.parse
是否有 markdown 残留
是否有测试代码污染
是否明显截断
```

这一层不产生正式 pass@1。

### 4.2 第二层：本地受限执行原型

在真正 Docker / EvalPlus 之前，可以做一个小型受限执行原型。

该原型需要至少包含：

```text
subprocess 隔离
timeout 限制
临时目录执行
禁用网络
限制单样本执行时间
执行后删除临时文件
只运行极少量样本
```

这一层只能用于工程调试，不能作为最终安全方案。

### 4.3 第三层：Docker / WSL / EvalPlus 正式执行

正式 pass@1 应该在隔离环境中运行。

推荐路线：

```text
Docker container
或 WSL isolated environment
或 EvalPlus 官方执行框架
```

正式执行时应该满足：

```text
每个样本独立执行
设置 timeout
限制内存
限制文件系统访问
不暴露主项目目录
不暴露密钥和环境变量
执行日志单独保存
失败样本单独记录
```

## 5. 后续代码分支推荐顺序

当前代码推理分支后续顺序建议为：

```text
1. HumanEval safe sample-only generation；
2. HumanEval sample inspection；
3. HumanEval static analysis；
4. EvalPlus / HumanEval+ / MBPP+ 接入调研；
5. sandboxed execution 原型；
6. 小样本正式 pass@1；
7. 根据错误构造 code SFT 数据；
8. 小规模 code SFT；
9. 数学 + 代码双任务最终总结。
```

## 6. 当前结论

当前不应该直接开启 `HF_ALLOW_CODE_EVAL=1`。

当前正确路线是：

```text
先扩大 safe sample-only generation
再做静态分析和语义分析
然后设计 sandboxed execution
最后再计算正式 pass@1
```

代码推理分支的核心经验是：

```text
static_clean 不等于 functional correctness。
```

这和数学推理分支的经验一致：

```text
format correctness 不等于 reasoning correctness。
```

因此后续代码推理优化不能只关注格式，而要引入：

```text
测试反馈
边界条件
隐藏用例
execution-based reward
sandboxed evaluation
```