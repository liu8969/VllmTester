import asyncio
import aiohttp
import json
import time
import argparse
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Union, Optional
from rich.progress import Progress
from rich.table import Table
from rich.console import Console

# 默认配置
DEFAULT_ENDPOINT = "http://localhost:8000/v1/completions"
DEFAULT_MODEL = "DEFAULT"
DEFAULT_HEADERS = {"Content-Type": "application/json"}

def create_output_directory(model_name: str) -> str:
    """创建输出目录，格式为: 模型名_年月日_时分秒"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dir_name = f"{model_name}_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def load_prompts_from_file(filename: str) -> List[str]:
    """从文件加载提示词列表"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Failed to load prompt file: {e}")
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
                f"[cyan]Concurrency {max_concurrent} - Sending requests...",
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
        print(f"💾 Results saved to: {filename}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

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
    
    if not results:  # 处理空结果的情况
        return stats
    
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
    stats_table = Table(title=f"📊 Concurrency {concurrency} Summary (Requests: {stats['total_requests']})", 
                       show_header=False, style="blue")
    stats_table.add_row("Total Requests", f"{stats['total_requests']}")
    stats_table.add_row("Successful", f"{stats['success_count']} ({stats['success_count']/stats['total_requests']*100:.1f}%)")
    stats_table.add_row("Failed", f"{stats['error_count']} ({stats['error_count']/stats['total_requests']*100:.1f}%)")
    stats_table.add_row("Total Time", f"{stats['total_time']:.2f} seconds")
    stats_table.add_row("Total Tokens", f"{stats['total_tokens']}")
    
    if stats['total_time'] > 0:
        stats_table.add_row("Request Throughput", f"{stats['requests_per_sec']:.2f} req/sec")
        if stats['total_tokens'] > 0:
            stats_table.add_row("Token Throughput", f"{stats['tokens_per_sec']:.2f} tokens/sec")
            stats_table.add_row("Per-request Token Throughput",
                              f"{stats['tokens_per_sec'] / concurrency:.2f} tokens/sec/conn" if concurrency > 0 else "N/A")
    
    if stats['success_count'] > 0:
        stats_table.add_row("Avg Tokens per Request",
                          f"{stats['avg_tokens_per_request']:.1f} tokens/req")
    
    # 延迟统计
    latency_table = Table(title="⏱ Latency Statistics", show_header=False, style="green")
    latency_table.add_row("Average Latency", f"{stats['avg_latency']:.3f} seconds")
    latency_table.add_row("Min Latency", f"{stats['min_latency']:.3f} seconds")
    latency_table.add_row("Max Latency", f"{stats['max_latency']:.3f} seconds")
    
    console.print(stats_table)
    console.print(latency_table)
    return stats

def generate_performance_plot(concurrency_levels: List[int], stats_list: List[Dict], output_file: str):
    """生成并发性能对比图表（英文版）"""
    plt.figure(figsize=(15, 10))
    
    # 准备数据
    tokens_per_sec = [s["tokens_per_sec"] for s in stats_list]
    per_request_tokens_per_sec =[s["tokens_per_sec"] / c for s, c in zip(stats_list, concurrency_levels)]
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
    output_dir: str,  # 新增参数：输出目录
    requests_per_concurrency: int,
    **request_params
) -> Dict:
    """运行指定并发级别的测试，请求数与并发数成正比"""
    # 计算该并发级别下实际的请求数量
    actual_num_requests = concurrency * requests_per_concurrency
    print(f"\n🔁 Starting test for concurrency: {concurrency} (Requests: {actual_num_requests})")
    
    # 准备该并发级别所需的提示词
    if len(prompts) < actual_num_requests:
        # 如果提示词不足，循环使用
        concurrency_prompts = (prompts * (actual_num_requests // len(prompts) + 1))[:actual_num_requests]
    else:
        concurrency_prompts = prompts[:actual_num_requests]
    
    # 使用Rich进度条
    try:
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
    
    # 计算统计信息 - 确保stats不为None
    stats = calculate_statistics(results, total_time)
    if stats is None:
        print(f"⚠️ Warning: Statistics calculation failed for concurrency {concurrency}")
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
    
    # 添加实际请求数到统计信息
    stats["actual_requests"] = actual_num_requests
    
    # 打印统计信息
    print_statistics(stats, concurrency)
    
    # 保存结果到输出目录
    output_file = os.path.join(output_dir, f"results_{model}_concurrency_{concurrency}.json")
    save_results_to_file(results, output_file)
    
    return stats

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="vLLM Concurrency Benchmark Tool")
    
    # 基本参数
    parser.add_argument("--num-requests", type=int, default=0, help="Total requests (overridden by --requests-per-concurrency)")
    parser.add_argument("--concurrency", type=str, default="1,2,4,8", help="Concurrency levels (e.g., '1,4,8' or '1-8')")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="API endpoint")
    parser.add_argument("--prompt-file", type=str, help="Path to file containing prompts (one per line)")
    parser.add_argument("--output-plot", type=str, default="performance_plot.png", help="Performance chart filename")
    
    # 新增参数：每个并发的基准请求数
    parser.add_argument("--requests-per-concurrency", type=int, default=10,
                        help="Base requests per concurrency level (actual = concurrency × this)")
    
    # 模型参数
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0-2)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling probability (0-1)")
    parser.add_argument("--stop", type=str, nargs="+", help="Stop sequences (e.g., '\\n' '###')")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="Topic repetition penalty (-2-2)")
    parser.add_argument("--frequency-penalty", type=float, default=0.0, help="Word repetition penalty (-2-2)")
    parser.add_argument("--best-of", type=int, default=1, help="Number of best results to return")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=1, help="Number of retries for failed requests")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_output_directory(args.model)
    print(f"📂 All results will be saved in: {output_dir}")
    
    # 计算最大并发级别所需的提示词数量
    if ',' in args.concurrency:
        concurrency_list = args.concurrency.split(',')
        max_concurrency = max(map(int, concurrency_list))
    elif '-' in args.concurrency:
        start, end = map(int, args.concurrency.split('-'))
        max_concurrency = end
    else:
        max_concurrency = int(args.concurrency)
    
    max_requests = max_concurrency * args.requests_per_concurrency
    
    # 准备提示词
    if args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
        if not prompts:
            print("⚠️ Using default prompts")
            prompts = [f"Test prompt #{i}" for i in range(max_requests)]
        else:
            # 如果文件中的提示词少于所需数量，循环使用
            if len(prompts) < max_requests:
                prompts = (prompts * (max_requests // len(prompts) + 1))[:max_requests]
            else:
                prompts = prompts[:max_requests]  # 只取前max_requests个
    else:
        prompts = [f"Test prompt #{i}" for i in range(max_requests)]
    
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
    
    print(f"🚀 Starting vLLM Concurrency Benchmark")
    print(f"├─ Model: {args.model}")
    print(f"├─ Endpoint: {args.endpoint}")
    print(f"├─ Base requests per concurrency: {args.requests_per_concurrency}")
    print(f"├─ Max requests (at highest concurrency): {max_requests}")
    print(f"├─ Concurrency levels: {concurrency_levels}")
    print(f"├─ Max tokens: {args.max_tokens}")
    print(f"├─ Temperature: {args.temperature}")
    print(f"├─ Top-p: {args.top_p}")
    print(f"├─ Timeout: {args.timeout} seconds")
    print(f"└─ Retries: {args.retries}")
    
    if args.stop:
        print(f"├─ Stop sequences: {args.stop}")
    if args.best_of > 1:
        print(f"├─ Best-of: {args.best_of}")
    
    print("\n📋 Sample prompts:")
    sample_count = min(3, len(prompts))
    for i, prompt in enumerate(prompts[:sample_count]):
        print(f"  {i+1}. {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    if len(prompts) > sample_count:
        print(f"  ... and {len(prompts)-sample_count} more")
    
    # 运行所有并发级别的测试
    all_stats = []
    for concurrency in concurrency_levels:
        stats = run_concurrency_test(
            prompts=prompts,
            concurrency=concurrency,
            model=args.model,
            endpoint=args.endpoint,
            output_dir=output_dir,  # 传入输出目录
            requests_per_concurrency=args.requests_per_concurrency,
            **request_params
        )
        all_stats.append(stats)
    
    # 生成性能对比图表并保存到输出目录
    plot_path = os.path.join(output_dir, args.output_plot)
    generate_performance_plot(concurrency_levels, all_stats, plot_path)
    
    # 打印最终摘要
    print("\n🎯 Benchmark completed! Performance summary:")
    for i, concurrency in enumerate(concurrency_levels):
        stats = all_stats[i]
        print(f"Concurrency {concurrency} (Requests: {stats['actual_requests']}):")
        print(f"  Token Throughput: {stats['tokens_per_sec']:.2f} tokens/sec")
        print(f"  Per-request Throughput: {stats['tokens_per_sec'] / concurrency:.2f} tokens/sec/conn")
        print(f"  Average Latency: {stats['avg_latency']:.3f} seconds")
        print(f"  Request Throughput: {stats['requests_per_sec']:.2f} requests/sec")
    
    # 保存汇总统计
    summary_file = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "model": args.model,
            "endpoint": args.endpoint,
            "concurrency_levels": concurrency_levels,
            "stats": all_stats
        }, f, indent=2)
    print(f"\n📊 Benchmark summary saved to: {summary_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
