
[English](https://github.com/liu8969/VllmTester/blob/main/README.en.md)
[中文](https://github.com/liu8969/VllmTester/blob/main/README.md)
# 功能特点
1. ⚡ 动态请求缩放：请求数 = 并发数 × 基准请求数
2. 📊 彩色终端统计报表：使用 Rich 库提供美观的统计输出
3. 📈 自动生成性能图表：直观展示吞吐量、延迟等关键指标
4. 🔁 自动重试机制：对失败请求自动重试
5. 💾 结果保存：将详细测试结果保存为 JSON 文件
6. 🚦 灵活并发级别测试：支持范围测试（如1-16）和指定级别测试

# 安装VllmTester

### 克隆仓库
    https://github.com/liu8969/VllmTester.git
    cd VllmTester
### 安装依赖
    pip install -r requirements.txt

# 准备问题集
将您的问题集保存为文本文件（如 `prompts.txt`），每行一个问题  

示例如下：
~~~
解释量子纠缠的概念
写一首关于秋天的诗
如何提高深度学习模型的准确率？
...
...
~~~
如果没有，我这里有一个,为你准备


# 参数说明：


参数|描述|默认值|
|-------|-------|-------|
--model|要测试的模型名称|"DEFAULT"|
--endpoint|API 端点| URL"http://localhost:8000/v1/completions "|
--concurrency|要测试的并发级别 |(如 "1,2,4" 或 "1-8")"1,2,4,8"|
--requests-per-concurrency|每个并发级别的基准请求数|10
--prompt-file|包含提示词的文件 (每行一个)|无
--max-tokens|每个响应的最大 token 数|256
--temperature|采样温度|0.7
--top-p|Top-p 采样值|0.95
--stop|停止序列|无
--presence-penalty|主题重复惩罚|0.0
--frequency-penalty|词语重复惩罚|0.0
--best-of|Best-of 采样|1
--timeout|请求超时时间 (秒)|120
--retries|失败请求重试次数|1
--output-plot|性能图表文件名|"performance_plot.png"



# 执行测试命令
使用以下命令执行测试（根据需求调整参数）：
~~~
 python3 VllmTester2.0.py   \
 --prompt-file prompts.txt \     
 --concurrency "1,2,4,8,16" \
 --requests-per-concurrency 4 \  
 --model Qwen3-32B-AWQ   \
 --endpoint http://localhost:8000/v1/completions \
 --max-tokens 512   \
 --temperature 0.7   \
 --top-p 0.95 \
 --stop "\n" "###" \
 --timeout 180   \
 --retries 2   \
 --output-plot qwen_performance.png    

~~~

# 输出结果示例
~~~ 
🚀 启动 vLLM 并发性能测试
├─ 模型: Qwen3-32B-AWQ
├─ 端点: http://localhost:8000/v1/completions
├─ 每个并发基准请求数: 4
├─ 最大请求数（最大并发时）: 64
├─ 并发级别: [1, 2, 4, 8, 16]
├─ 最大token数: 512
├─ 温度: 0.7
├─ Top-p: 0.95
├─ 超时时间: 180秒
└─ 重试次数: 2

📋 示例提示:
  1. 解释量子纠缠的概念
  2. 写一首关于秋天的诗
  3. 如何提高深度学习模型的准确率？
  ... 和另外 61 个提示

🔁 开始测试并发数: 1 (请求数: 4)
并发 1 - 发送请求... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
         📊 并发 1 性能摘要 (请求数: 4)          
┌─────────────────────────┬─────────────────────┐
│ 总请求数                │ 4                   │
│ 成功请求                │ 4 (100.0%)          │
│ 失败请求                │ 0 (0.0%)            │
│ 总耗时                  │ 40.34秒             │
│ 总生成token数           │ 1332                │
│ 请求吞吐量              │ 0.10 请求/秒        │
│ Token吞吐量             │ 33.02 token/秒      │
│ 单个请求平均token吞吐量 │ 33.02 token/秒/连接 │
│ 平均生成token数         │ 333.0 token/请求    │
└─────────────────────────┴─────────────────────┘
      ⏱ 延迟统计       
┌──────────┬──────────┐
│ 平均延迟 │ 20.855秒 │
│ 最小延迟 │ 3.127秒  │
│ 最大延迟 │ 40.333秒 │
└──────────┴──────────┘
💾 结果已保存到: results_Qwen3-32B-AWQ_concurrency_1.json

...
...
...
...
💾 结果已保存到: results_Qwen3-32B-AWQ_concurrency_16.json
📈 Performance chart (PDF) saved to: qwen_performance.pdf
📈 Performance chart (PNG) saved to: qwen_performance.png

🎯 测试完成! 性能摘要:
并发 1 (请求数: 4):
  Token吞吐量: 33.02 token/秒
  单个请求吞吐量: 33.02 token/秒/连接
  平均延迟: 20.855秒
  请求吞吐量: 0.10 请求/秒
并发 2 (请求数: 8):
  Token吞吐量: 62.45 token/秒
  单个请求吞吐量: 31.22 token/秒/连接
  平均延迟: 31.200秒
  请求吞吐量: 0.17 请求/秒
并发 4 (请求数: 16):
  Token吞吐量: 102.68 token/秒
  单个请求吞吐量: 25.67 token/秒/连接
  平均延迟: 27.301秒
  请求吞吐量: 0.33 请求/秒
并发 8 (请求数: 32):
  Token吞吐量: 191.88 token/秒
  单个请求吞吐量: 23.99 token/秒/连接
  平均延迟: 25.755秒
  请求吞吐量: 0.57 请求/秒
并发 16 (请求数: 64):
  Token吞吐量: 293.01 token/秒
  单个请求吞吐量: 18.31 token/秒/连接
  平均延迟: 31.991秒
  请求吞吐量: 0.93 请求/秒

~~~
![输出图表示例](https://github.com/liu8969/VllmTester/blob/main/performance_plot.png)
