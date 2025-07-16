
# Features
1. ⚡ Dynamic request scaling: Request count = Concurrency × Base requests
2. 📊 Colorful terminal statistics: Beautiful statistical output using Rich library
3. 📈 Automatic performance charts: Visualizes key metrics like throughput and latency
4. 🔁 Automatic retry mechanism: Automatically retries failed requests
5. 💾 Result saving: Saves detailed test results as JSON files
6. 🚦 Flexible concurrency testing: Supports range testing (e.g., 1-16) and specified levels

# Installation
### Clone repository
```bash
git clone https://github.com/liu8969/VllmTester.git
cd VllmTester
```
### Install dependencies
    pip install -r requirements.txt

# Preparing Prompt Sets
Save your prompt set as a text file (e.g., `prompts.txt`), with one prompt per line:

Example:
~~~
Explain the concept of quantum entanglement
Write a poem about autumn
How to improve the accuracy of deep learning models?
...
...
~~~

# Parameters
|Parameter|Description|Default|
|-------|-------|-------|
--model|Model name to test|"DEFAULT"|
--endpoint|API endpoint|"http://localhost:8000/v1/completions "|
--concurrency|Concurrency levels to test (e.g. "1,2,4" or "1-8")|"1,2,4,8"
--requests-per-concurrency|Base request count per|concurrency level|10
--prompt-file|File containing prompts (one per line)|None
--max-tokens|Maximum tokens per response|256
--temperature|Sampling temperature|0.7|
--top-p|Top-p sampling value|0.95
--stop|Stop sequences|None
--presence-penalty|Topic repetition penalty|0.0
--frequency-penalty|Token repetition penalty|0.0
--best-of|Best-of sampling|1
--timeout|Request timeout (seconds)|120
--retries|Number of retries for failed requests|1
--output-plot|Performance plot filename|"performance_plot.png"

# Running Tests

Execute tests with the following command (adjust parameters as needed):
~~~
python3 VllmTester2.0.py \
  --prompt-file prompts.txt \
  --concurrency "1,2,4,8,16" \
  --requests-per-concurrency 4 \
  --model Qwen3-32B-AWQ \
  --max-tokens 512 \
  --temperature 0.7 \
  --timeout 180 \
  --retries 2 \
  --output-plot qwen_performance.png
  ~~~
  # Sample Output
  ~~~
 🚀 Starting vLLM concurrency performance test
├─ Model: Qwen3-32B-AWQ
├─ Endpoint: http://localhost:8000/v1/completions
├─ Base requests per concurrency: 4
├─ Maximum requests (at peak concurrency): 64
├─ Concurrency levels: [1, 2, 4, 8, 16]
├─ Max tokens: 512
├─ Temperature: 0.7
├─ Top-p: 0.95
├─ Timeout: 180 seconds
└─ Retries: 2
📋 Sample prompts:
  1. Explain the concept of quantum entanglement
  2. Write a poem about autumn
  3. How to improve deep learning model accuracy?
  ... plus 61 more prompts
🔁 Starting test with concurrency: 1 (requests: 4)
Concurrency 1 - Sending requests... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
         📊 Concurrency 1 Performance Summary (Requests: 4)          
┌─────────────────────────┬─────────────────────┐
│ Total Requests          │ 4                   │
│ Successful Requests     │ 4 (100.0%)          │
│ Failed Requests         │ 0 (0.0%)            │
│ Total Duration          │ 40.34s              │
│ Total Tokens Generated  │ 1332                │
│ Request Throughput      │ 0.10 requests/sec   │
│ Token Throughput        │ 33.02 tokens/sec    │
│ Avg Token Throughput    │ 33.02 tokens/sec/conn │
│ Avg Tokens per Request  │ 333.0 tokens/request│
└─────────────────────────┴─────────────────────┘
      ⏱ Latency Statistics       
┌──────────┬──────────┐
│ Avg Latency │ 20.855s │
│ Min Latency │ 3.127s  │
│ Max Latency │ 40.333s │
└──────────┴──────────┘
💾 Results saved to: results_Qwen3-32B-AWQ_concurrency_1.json
...
...
💾 Results saved to: results_Qwen3-32B-AWQ_concurrency_16.json
📈 Performance chart (PDF) saved to: qwen_performance.pdf
📈 Performance chart (PNG) saved to: qwen_performance.png
🎯 Test completed! Performance summary:
Concurrency 1 (Requests: 4):
  Token Throughput: 33.02 tokens/sec
  Per-request Throughput: 33.02 tokens/sec/conn
  Avg Latency: 20.855s
  Request Throughput: 0.10 requests/sec
Concurrency 2 (Requests: 8):
  Token Throughput: 62.45 tokens/sec
  Per-request Throughput: 31.22 tokens/sec/conn
  Avg Latency: 31.200s
  Request Throughput: 0.17 requests/sec
Concurrency 4 (Requests: 16):
  Token Throughput: 102.68 tokens/sec
  Per-request Throughput: 25.67 tokens/sec/conn
  Avg Latency: 27.301s
  Request Throughput: 0.33 requests/sec
Concurrency 8 (Requests: 32):
  Token Throughput: 191.88 tokens/sec
  Per-request Throughput: 23.99 tokens/sec/conn
  Avg Latency: 25.755s
  Request Throughput: 0.57 requests/sec
Concurrency 16 (Requests: 64):
  Token Throughput: 293.01 tokens/sec
  Per-request Throughput: 18.31 tokens/sec/conn
  Avg Latency: 31.991s
  Request Throughput: 0.93 requests/sec
  ~~~
