import asyncio
import aiohttp
import json
import time
import argparse
import math
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from rich.progress import Progress
from rich.table import Table
from rich.console import Console

# 默认配置
DEFAULT_ENDPOINT = "http://localhost:8000/v1/completions"
DEFAULT_MODEL = "DEFAULT"
DEFAULT_HEADERS = {"Content-Type": "application/json"}

def load_prompts_from_file(filename: str) -> List[str]:
    """从文件加载提示词列表"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"❌ 无法加载提示词文件: {e}")
        return []

async def send_vllm_request(
    session: aiohttp.ClientSession,
    prompt: str,
    request_id: int,
    model: str,
    endpoint: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    stop: Optional[Union[str, List[str]]] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logprobs: Optional[int] = None,
    best_of: int = 1,
    logit_bias: Optional[Dict[int, float]] = None,
    timeout: int = 120,
    retries: int = 1
) -> Dict:
    """发送请求到 vLLM 服务，包含客户端计时和重试机制"""
    start_time = time.perf_counter()
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False
    }
    if stop: payload["stop"] = stop
    if presence_penalty != 0.0: payload["presence_penalty"] = presence_penalty
    if frequency_penalty != 0.0: payload["frequency_penalty"] = frequency_penalty
    if logprobs: payload["logprobs"] = logprobs
    if best_of > 1: payload["best_of"] = best_of
    if logit_bias: payload["logit_bias"] = logit_bias
    
    for attempt in range(retries + 1):
        current_timeout = timeout * math.pow(1.5, attempt)
        try:
            async with session.post(
                endpoint,
                json=payload,
                headers=DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=current_timeout)
            ) as response:
                client_latency = time.perf_counter() - start_time
                process_time = float(response.headers.get("X-Process-Time", 0))
                try:
                    response_data = await response.json()
                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                    text = await response.text()
                    return {
                        "client_id": request_id,
                        "vllm_id": response.headers.get("X-Request-ID", "N/A"),
                        "error": f"Invalid JSON response: {text[:200]}",
                        "prompt": prompt,
                        "latency": client_latency,
                        "client_latency": client_latency,
                        "server_latency": process_time,
                        "attempts": attempt + 1
                    }
                if response.status == 200 and "choices" in response_data:
                    return {
                        "client_id": request_id,
                        "vllm_id": response.headers.get("X-Request-ID", "N/A"),
                        "prompt": prompt,
                        "response": response_data["choices"][0]["text"].strip(),
                        "latency": process_time if process_time > 0 else client_latency,
                        "tokens_used": response_data.get("usage", {}).get("total_tokens", 0),
                        "details": response_data,
                        "client_latency": client_latency,
                        "server_latency": process_time,
                        "attempts": attempt + 1
                    }
                else:
                    error_msg = response_data.get("error", {}).get("message", "Unknown error")
                    return {
                        "client_id": request_id,
                        "vllm_id": response.headers.get("X-Request-ID", "N/A"),
                        "error": f"HTTP {response.status}: {error_msg}",
                        "prompt": prompt,
                        "latency": client_latency,
                        "client_latency": client_latency,
                        "server_latency": process_time,
                        "attempts": attempt + 1
                    }
        except asyncio.TimeoutError:
            client_latency = time.perf_counter() - start_time
            if attempt < retries:
                continue
            return {
                "client_id": request_id,
                "error": f"Request timed out after {current_timeout:.1f}s",
                "prompt": prompt,
                "latency": client_latency,
                "client_latency": client_latency,
                "server_latency": 0.0,
                "attempts": attempt + 1
            }
        except Exception as e:
            client_latency = time.perf_counter() - start_time
            return {
                "client_id": request_id,
                "error": str(e),
                "prompt": prompt,
                "latency": client_latency,
                "client_latency": client_latency,
                "server_latency": 0.0,
                "attempts": attempt + 1
            }

async def batch_request_vllm(
    prompts: List[str],
    max_concurrent: int,
    model: str,
    endpoint: str,
    progress: Optional[Progress] = None,
    **request_params
) -> List[Dict]:
    """批量发送自定义请求"""
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        task_progress = None
        if progress:
            task_progress = progress.add_task(
                f"[cyan]并发 {max_concurrent} - 发送请求...",
                total=len(prompts))
        for idx, prompt in enumerate(prompts):
            task = asyncio.create_task(
                send_vllm_request(
                    session=session,
                    prompt=prompt,
                    request_id=idx,
                    model=model,
                    endpoint=endpoint,
                    **request_params
                )
            )
            if progress:
                task.add_done_callback(lambda _: progress.update(task_progress, advance=1))
            tasks.append(task)
        return await asyncio.gather(*tasks)

def save_results_to_file(results: List[Dict], filename: str):
    """保存结果到JSON文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 结果已保存到: {filename}")
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")

def calculate_statistics(results: List[Dict], total_time: float) -> Dict:
    """计算性能统计数据"""
    stats = {
        "total_requests": len(results),
        "total_time": total_time,
        "success_count": 0,
        "error_count": 0,
        "total_tokens": 0,
        "requests_per_sec": 0,
        "tokens_per_sec": 0,
        "per_request_tokens_per_sec": 0,
        "avg_latency": 0,
        "min_latency": 0,
        "max_latency": 0,
        "avg_tokens_per_request": 0
    }
    success_results = [r for r in results if "response" in r]
    error_results = [r for r in results if "error" in r]
    stats["success_count"] = len(success_results)
    stats["error_count"] = len(error_results)
    stats["total_tokens"] = sum(r.get("tokens_used", 0) for r in success_results)
    
    # 计算吞吐量
    if total_time > 0:
        stats["requests_per_sec"] = len(results) / total_time
        stats["tokens_per_sec"] = stats["total_tokens"] / total_time
    
    # 计算延迟
    latencies = [r.get("client_latency", 0) for r in results]
    if latencies:
        stats["avg_latency"] = sum(latencies) / len(latencies)
        stats["min_latency"] = min(latencies)
        stats["max_latency"] = max(latencies)
    
    # 计算每个请求的token指标
    if stats["success_count"] > 0:
        stats["avg_tokens_per_request"] = stats["total_tokens"] / stats["success_count"]
    
    return stats

def print_statistics(stats: Dict, concurrency: int):
    """打印性能统计信息"""
    console = Console()
    # 创建统计表格
    stats_table = Table(title=f"📊 并发 {concurrency} 性能摘要 (请求数: {stats['total_requests']})", show_header=False, style="blue")
    stats_table.add_row("总请求数", f"{stats['total_requests']}")
    stats_table.add_row("成功请求", f"{stats['success_count']} ({stats['success_count']/stats['total_requests']*100:.1f}%)")
    stats_table.add_row("失败请求", f"{stats['error_count']} ({stats['error_count']/stats['total_requests']*100:.1f}%)")
    stats_table.add_row("总耗时", f"{stats['total_time']:.2f}秒")
    stats_table.add_row("总生成token数", f"{stats['total_tokens']}")
    
    if stats['total_time'] > 0:
        stats_table.add_row("请求吞吐量", f"{stats['requests_per_sec']:.2f} 请求/秒")
        if stats['total_tokens'] > 0:
            stats_table.add_row("Token吞吐量", f"{stats['tokens_per_sec']:.2f} token/秒")
            stats_table.add_row("单个请求平均token吞吐量",
                              f"{stats['tokens_per_sec'] / concurrency:.2f} token/秒/连接" if concurrency > 0 else "N/A")
    
    if stats['success_count'] > 0:
        stats_table.add_row("平均生成token数",
                          f"{stats['avg_tokens_per_request']:.1f} token/请求")
    
    # 延迟统计
    latency_table = Table(title="⏱ 延迟统计", show_header=False, style="green")
    latency_table.add_row("平均延迟", f"{stats['avg_latency']:.3f}秒")
    latency_table.add_row("最小延迟", f"{stats['min_latency']:.3f}秒")
    latency_table.add_row("最大延迟", f"{stats['max_latency']:.3f}秒")
    
    console.print(stats_table)
    console.print(latency_table)
    return stats

def generate_performance_plot(concurrency_levels: List[int], stats_list: List[Dict], output_file: str):
    """生成并发性能对比图表（英文版）"""
    plt.figure(figsize=(15, 10))
    
    # 准备数据
    tokens_per_sec = [s["tokens_per_sec"] for s in stats_list]
    per_request_tokens_per_sec = [s["tokens_per_sec"] / c for s, c in zip(stats_list, concurrency_levels)]
    avg_latency = [s["avg_latency"] for s in stats_list]
    
    # 设置图表标题和标签（英文）
    title1 = 'Total Token Throughput vs Concurrency'
    title2 = 'Per-Request Token Throughput vs Concurrency'
    title3 = 'Average Latency vs Concurrency'
    title4 = 'Request Throughput vs Concurrency'
    xlabel = 'Concurrency'
    ylabel1 = 'Tokens/s'
    ylabel2 = 'Tokens/s per connection'
    ylabel3 = 'Seconds'
    ylabel4 = 'Requests/s'
    
    plt.subplot(2, 2, 1)
    plt.plot(concurrency_levels, tokens_per_sec, 'o-', color='#1f77b4')
    plt.title(title1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel1)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.plot(concurrency_levels, per_request_tokens_per_sec, 'o-', color='#ff7f0e')
    plt.title(title2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel2)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 3)
    plt.plot(concurrency_levels, avg_latency, 'o-', color='#2ca02c')
    plt.title(title3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel3)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    plt.plot(concurrency_levels, [s["requests_per_sec"] for s in stats_list], 'o-', color='#d62728')
    plt.title(title4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel4)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 尝试保存为PDF格式
    pdf_saved = False
    if output_file.endswith('.png'):
        pdf_file = output_file.replace('.png', '.pdf')
        try:
            plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
            print(f"📈 Performance chart (PDF) saved to: {pdf_file}")
            pdf_saved = True
        except Exception as e:
            print(f"⚠️ Failed to save PDF chart: {e}")
    
    # 保存为PNG格式
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📈 Performance chart (PNG) saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save chart: {e}")
    
    # 如果没有成功保存任何图表，尝试使用纯英文保存
    if not pdf_saved and not os.path.exists(output_file):
        try:
            # 强制使用英文
            plt.subplot(2, 2, 1).set_title(title1)
            plt.subplot(2, 2, 1).set_xlabel(xlabel)
            plt.subplot(2, 2, 1).set_ylabel(ylabel1)
            
            plt.subplot(2, 2, 2).set_title(title2)
            plt.subplot(2, 2, 2).set_xlabel(xlabel)
            plt.subplot(2, 2, 2).set_ylabel(ylabel2)
            
            plt.subplot(2, 2, 3).set_title(title3)
            plt.subplot(2, 2, 3).set_xlabel(xlabel)
            plt.subplot(2, 2, 3).set_ylabel(ylabel3)
            
            plt.subplot(2, 2, 4).set_title(title4)
            plt.subplot(2, 2, 4).set_xlabel(xlabel)
            plt.subplot(2, 2, 4).set_ylabel(ylabel4)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📈 English performance chart saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save chart in any format: {e}")
        finally:
            plt.close()

def run_concurrency_test(
    prompts: List[str],
    concurrency: int,
    model: str,
    endpoint: str,
    requests_per_concurrency: int,  # 新增参数：每个并发的基准请求数
    **request_params
) -> Dict:
    """运行指定并发级别的测试，请求数与并发数成正比"""
    # 计算该并发级别下实际的请求数量
    actual_num_requests = concurrency * requests_per_concurrency
    print(f"\n🔁 开始测试并发数: {concurrency} (请求数: {actual_num_requests})")
    
    # 准备该并发级别所需的提示词
    if len(prompts) < actual_num_requests:
        # 如果提示词不足，循环使用
        concurrency_prompts = (prompts * (actual_num_requests // len(prompts) + 1))[:actual_num_requests]
    else:
        concurrency_prompts = prompts[:actual_num_requests]
    
    # 使用Rich进度条
    try:
        from rich.progress import Progress
        progress = Progress()
    except ImportError:
        progress = None
    
    # 运行测试
    start_time = time.time()
    if progress:
        with progress:
            results = asyncio.run(
                batch_request_vllm(
                    prompts=concurrency_prompts,
                    max_concurrent=concurrency,
                    model=model,
                    endpoint=endpoint,
                    progress=progress,
                    **request_params
                )
            )
    else:
        results = asyncio.run(
            batch_request_vllm(
                prompts=concurrency_prompts,
                max_concurrent=concurrency,
                model=model,
                endpoint=endpoint,
                progress=None,
                **request_params
            )
        )
    
    total_time = time.time() - start_time
    
    # 计算统计信息
    stats = calculate_statistics(results, total_time)
    stats["actual_requests"] = actual_num_requests  # 记录实际请求数
    
    # 打印统计信息
    print_statistics(stats, concurrency)
    
    # 保存结果
    output_file = f"results_{model}_concurrency_{concurrency}.json"
    save_results_to_file(results, output_file)
    
    return stats

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="vLLM 并发性能测试工具")
    
    # 基本参数
    parser.add_argument("--num-requests", type=int, default=0, help="请求总数 (将被 --requests-per-concurrency 覆盖)")
    parser.add_argument("--concurrency", type=str, default="1,2,4,8", help="并发数测试范围 (如: '1,4,8' 或 '1-8')")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="API端点")
    parser.add_argument("--prompt-file", type=str, help="包含提示词的文件路径（每行一个）")
    parser.add_argument("--output-plot", type=str, default="performance_plot.png", help="性能图表文件名")
    
    # 新增参数：每个并发的基准请求数
    parser.add_argument("--requests-per-concurrency", type=int, default=10,
                        help="每个并发级别的基准请求数 (实际请求数 = 并发数 × 此值)")
    
    # 模型参数
    parser.add_argument("--max-tokens", type=int, default=256, help="每个响应的最大token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度 (0-2)")
    parser.add_argument("--top-p", type=float, default=0.95, help="核心采样概率 (0-1)")
    parser.add_argument("--stop", type=str, nargs="+", help="停止序列 (例如 '\\n' '###')")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="主题重复惩罚 (-2-2)")
    parser.add_argument("--frequency-penalty", type=float, default=0.0, help="词语重复惩罚 (-2-2)")
    parser.add_argument("--best-of", type=int, default=1, help="返回最佳结果的数量")
    parser.add_argument("--timeout", type=int, default=120, help="单个请求超时时间（秒）")
    parser.add_argument("--retries", type=int, default=1, help="失败请求重试次数")
    
    args = parser.parse_args()
    
    # 计算最大并发级别所需的提示词数量
    max_concurrency = max(
        [int(x) for x in args.concurrency.split(',')] 
        if ',' in args.concurrency else 
        [int(args.concurrency)]
    )
    max_requests = max_concurrency * args.requests_per_concurrency
    
    # 准备提示词
    if args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
        if not prompts:
            print("⚠️ 使用默认提示词")
            prompts = [f"测试提示 #{i}" for i in range(max_requests)]
        else:
            # 如果文件中的提示词少于所需数量，循环使用
            if len(prompts) < max_requests:
                prompts = (prompts * (max_requests // len(prompts) + 1))[:max_requests]
            else:
                prompts = prompts[:max_requests]  # 只取前max_requests个
    else:
        prompts = [f"测试提示 #{i}" for i in range(max_requests)]
    
    # 准备请求参数
    request_params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stop": args.stop,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "best_of": args.best_of,
        "timeout": args.timeout,
        "retries": args.retries
    }
    
    # 解析并发数范围
    if '-' in args.concurrency:
        # 格式: 1-8
        start, end = map(int, args.concurrency.split('-'))
        concurrency_levels = list(range(start, end+1))
    elif ',' in args.concurrency:
        # 格式: 1,2,4,8
        concurrency_levels = list(map(int, args.concurrency.split(',')))
    else:
        # 单个值
        concurrency_levels = [int(args.concurrency)]
    
    print(f"🚀 启动 vLLM 并发性能测试")
    print(f"├─ 模型: {args.model}")
    print(f"├─ 端点: {args.endpoint}")
    print(f"├─ 每个并发基准请求数: {args.requests_per_concurrency}")
    print(f"├─ 最大请求数（最大并发时）: {max_requests}")
    print(f"├─ 并发级别: {concurrency_levels}")
    print(f"├─ 最大token数: {args.max_tokens}")
    print(f"├─ 温度: {args.temperature}")
    print(f"├─ Top-p: {args.top_p}")
    print(f"├─ 超时时间: {args.timeout}秒")
    print(f"└─ 重试次数: {args.retries}")
    
    if args.stop:
        print(f"├─ 停止序列: {args.stop}")
    if args.best_of > 1:
        print(f"├─ Best-of: {args.best_of}")
    
    print("\n📋 示例提示:")
    sample_count = min(3, len(prompts))
    for i, prompt in enumerate(prompts[:sample_count]):
        print(f"  {i+1}. {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    if len(prompts) > sample_count:
        print(f"  ... 和另外 {len(prompts)-sample_count} 个提示")
    
    # 运行所有并发级别的测试
    all_stats = []
    for concurrency in concurrency_levels:
        stats = run_concurrency_test(
            prompts=prompts,
            concurrency=concurrency,
            model=args.model,
            endpoint=args.endpoint,
            requests_per_concurrency=args.requests_per_concurrency,
            **request_params
        )
        all_stats.append(stats)
    
    # 生成性能对比图表
    generate_performance_plot(concurrency_levels, all_stats, args.output_plot)
    
    # 打印最终摘要
    print("\n🎯 测试完成! 性能摘要:")
    for i, concurrency in enumerate(concurrency_levels):
        stats = all_stats[i]
        print(f"并发 {concurrency} (请求数: {stats['actual_requests']}):")
        print(f"  Token吞吐量: {stats['tokens_per_sec']:.2f} token/秒")
        print(f"  单个请求吞吐量: {stats['tokens_per_sec'] / concurrency:.2f} token/秒/连接")
        print(f"  平均延迟: {stats['avg_latency']:.3f}秒")
        print(f"  请求吞吐量: {stats['requests_per_sec']:.2f} 请求/秒")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n操作被用户中断")