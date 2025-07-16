import asyncio
import aiohttp
import json
import time
import argparse
import math
import os
import matplotlib.pyplot as plt
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
                total=len(prompts)
            )
        
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
    stats_table = Table(title=f"📊 并发 {concurrency} 性能摘要", show_header=False, style="blue")
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
    """生成并发性能对比图表"""
    # 设置支持中文的字体
    try:
        # 尝试使用系统中已安装的中文字体
        plt.rcParams['font.sans-serif'] = [
            'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 
            'WenQuanYi Zen Hei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif'
        ]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 测试中文显示
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "测试", fontsize=12)
        fig.canvas.draw()
        plt.close(fig)
        use_chinese = True
    except Exception as e:
        print(f"⚠️ 无法加载中文字体: {e}")
        use_chinese = False
        print("⚠️ 将使用英文标题")
    
    plt.figure(figsize=(15, 10))
    
    # 准备数据
    tokens_per_sec = [s["tokens_per_sec"] for s in stats_list]
    per_request_tokens_per_sec = [s["tokens_per_sec"] / c for s, c in zip(stats_list, concurrency_levels)]
    avg_latency = [s["avg_latency"] for s in stats_list]
    
    # 创建图表 - 根据字体支持情况选择标题语言
    if use_chinese:
        title1 = '总Token吞吐量 vs 并发数'
        title2 = '单个请求Token吞吐量 vs 并发数'
        title3 = '平均延迟 vs 并发数'
        title4 = '请求吞吐量 vs 并发数'
        xlabel = '并发数'
        ylabel1 = 'Token/秒'
        ylabel2 = 'Token/秒/连接'
        ylabel3 = '秒'
        ylabel4 = '请求/秒'
    else:
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
    
    # 尝试保存为PDF格式（通常对字体支持更好）
    pdf_saved = False
    if output_file.endswith('.png'):
        pdf_file = output_file.replace('.png', '.pdf')
        try:
            plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
            print(f"📈 性能图表(PDF)已保存到: {pdf_file}")
            pdf_saved = True
        except Exception as e:
            print(f"⚠️ 无法保存PDF图表: {e}")
    
    # 保存为PNG格式
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📈 性能图表(PNG)已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 保存图表失败: {e}")
    
    # 如果没有成功保存任何图表，尝试使用纯英文保存
    if not pdf_saved and not os.path.exists(output_file):
        try:
            # 强制使用英文
            title1 = 'Total Token Throughput vs Concurrency'
            title2 = 'Per-Request Token Throughput vs Concurrency'
            title3 = 'Average Latency vs Concurrency'
            title4 = 'Request Throughput vs Concurrency'
            xlabel = 'Concurrency'
            ylabel1 = 'Tokens/s'
            ylabel2 = 'Tokens/s per connection'
            ylabel3 = 'Seconds'
            ylabel4 = 'Requests/s'
            
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
            print(f"📈 英文版性能图表已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 无法保存任何格式的图表: {e}")
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
    # ... 其余代码保持不变 ...
    
    # 运行测试
    start_time = time.time()
    # ... 调用batch_request_vllm的代码保持不变 ...
    
    # 计算统计信息
    stats = calculate_statistics(results, total_time)
    stats["actual_requests"] = actual_num_requests  # 记录实际请求数
    
    # ... 其余代码保持不变 ...
    return stats

def main():
    # ... 参数解析部分保持不变 ...
    
    # 添加新参数：每个并发的基准请求数
    parser.add_argument("--requests-per-concurrency", type=int, default=10,
                        help="每个并发级别的基准请求数 (实际请求数 = 并发数 × 此值)")
    
    args = parser.parse_args()
    
    # 准备提示词（现在加载所有提示词，每个并发级别从中选取所需数量）
    if args.prompt_file:
        all_prompts = load_prompts_from_file(args.prompt_file)
        if not all_prompts:
            print("⚠️ 使用默认提示词")
            all_prompts = [f"测试提示 #{i}" for i in range(1000)]  # 生成足够多的默认提示词
    else:
        all_prompts = [f"测试提示 #{i}" for i in range(1000)]  # 生成足够多的默认提示词
    
    # ... 其余参数准备保持不变 ...
    
    # 解析并发数范围
    # ... 保持不变 ...
    
    print(f"🚀 启动 vLLM 并发性能测试")
    print(f"├─ 模型: {args.model}")
    print(f"├─ 端点: {args.endpoint}")
    print(f"├─ 每个并发基准请求数: {args.requests_per_concurrency}")
    print(f"├─ 并发级别: {concurrency_levels}")
    # ... 其余打印保持不变 ...
    
    # 运行所有并发级别的测试
    all_stats = []
    for concurrency in concurrency_levels:
        stats = run_concurrency_test(
            prompts=all_prompts,  # 传入所有可用的提示词
            concurrency=concurrency,
            model=args.model,
            endpoint=args.endpoint,
            requests_per_concurrency=args.requests_per_concurrency,  # 传入新参数
            **request_params
        )
        all_stats.append(stats)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n操作被用户中断")