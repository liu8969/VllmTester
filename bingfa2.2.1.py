import asyncio
import aiohttp
import json
import time
import argparse
import math
from typing import List, Dict, Union, Optional
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
    timeout: int = 120,  # 增加默认超时时间
    retries: int = 1     # 添加重试机制
) -> Dict:
    """发送请求到 vLLM 服务，包含客户端计时和重试机制"""
    start_time = time.perf_counter()
    # 构建请求负载
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False
    }
    
    # 添加可选参数
    if stop:
        payload["stop"] = stop
    if presence_penalty != 0.0:
        payload["presence_penalty"] = presence_penalty
    if frequency_penalty != 0.0:
        payload["frequency_penalty"] = frequency_penalty
    if logprobs:
        payload["logprobs"] = logprobs
    if best_of > 1:
        payload["best_of"] = best_of
    if logit_bias:
        payload["logit_bias"] = logit_bias
    
    # 请求尝试循环
    for attempt in range(retries + 1):
        current_timeout = timeout * math.pow(1.5, attempt)  # 指数退避
        try:
            async with session.post(
                endpoint,
                json=payload,
                headers=DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=current_timeout)
            ) as response:
                client_latency = time.perf_counter() - start_time
                process_time = float(response.headers.get("X-Process-Time", 0))
                
                # 尝试解析JSON响应
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
                continue  # 重试
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
    progress: Progress,  # 进度条对象
    **request_params
) -> List[Dict]:
    """批量发送自定义请求"""
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        task_progress = progress.add_task("[cyan]发送请求...", total=len(prompts))
        
        # 创建所有任务
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

def print_statistics(results: List[Dict], total_time: float):
    """增强统计信息显示，包含超时分析"""
    console = Console()
    
    # 基本统计
    success_results = [r for r in results if "response" in r]
    error_results = [r for r in results if "error" in r]
    success_count = len(success_results)
    error_count = len(error_results)
    total_tokens = sum(r.get("tokens_used", 0) for r in success_results)
    
    # 延迟统计
    latencies = [r.get("client_latency", 0) for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    
    # 创建统计表格
    stats_table = Table(title="📊 性能统计摘要", show_header=False, style="blue")
    stats_table.add_row("总请求数", f"{len(results)}")
    stats_table.add_row("成功请求", f"{success_count} ({success_count/len(results)*100:.1f}%)")
    stats_table.add_row("失败请求", f"{error_count} ({error_count/len(results)*100:.1f}%)")
    stats_table.add_row("总耗时", f"{total_time:.2f}秒")
    stats_table.add_row("总生成token数", f"{total_tokens}")
    stats_table.add_row("请求吞吐量", f"{len(results)/total_time:.2f} 请求/秒")
    if total_tokens > 0:
        stats_table.add_row("Token吞吐量", f"{total_tokens/total_time:.2f} token/秒")
    
    # 延迟统计表格
    latency_table = Table(title="⏱ 延迟统计", show_header=False, style="green")
    latency_table.add_row("平均延迟", f"{avg_latency:.3f}秒")
    latency_table.add_row("最小延迟", f"{min_latency:.3f}秒")
    latency_table.add_row("最大延迟", f"{max_latency:.3f}秒")
    
    # 错误摘要表格
    if error_results:
        error_table = Table(title="❌ 错误摘要", style="red")
        error_table.add_column("错误类型")
        error_table.add_column("次数")
        
        error_types = {}
        for err in error_results:
            error_msg = err['error']
            # 简化长错误消息
            if len(error_msg) > 50:
                error_msg = error_msg.split(':')[0] + "..." if ':' in error_msg else error_msg[:50] + "..."
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
        
        for err_type, count in error_types.items():
            error_table.add_row(err_type, str(count))
    
    # 打印所有表格
    console.print(stats_table)
    console.print(latency_table)
    if error_results:
        console.print(error_table)
    
    # 超时请求详细分析
    timeout_results = [r for r in results if "timed out" in r.get("error", "")]
    if timeout_results:
        avg_timeout = sum(r['client_latency'] for r in timeout_results) / len(timeout_results)
        timeout_table = Table(title="⏰ 超时请求分析", style="yellow")
        timeout_table.add_column("尝试次数")
        timeout_table.add_column("平均等待时间")
        timeout_table.add_column("数量")
        timeout_table.add_column("占比")
        
        attempts = {}
        for r in timeout_results:
            att = r.get("attempts", 1)
            attempts[att] = attempts.get(att, 0) + 1
        
        for att, count in sorted(attempts.items()):
            timeout_table.add_row(
                str(att),
                f"{avg_timeout:.1f}s",
                str(count),
                f"{count/len(timeout_results)*100:.1f}%"
            )
        console.print(timeout_table)
        
        # 超时请求建议
        console.print(
            f"\n💡 建议: 当前超时时间可能不足，请考虑:\n"
            f"  - 增加 --timeout 参数值 (当前: {timeout_results[0]['error'].split('after')[-1]})\n"
            f"  - 减少 --max-tokens 参数值\n"
            f"  - 优化模型部署配置", 
            style="bold yellow"
        )
    
    # 延迟分布统计
    if success_results:
        latencies = [r["client_latency"] for r in success_results]
        if latencies:
            latencies_sorted = sorted(latencies)
            p90 = latencies_sorted[int(len(latencies_sorted) * 0.90)]
            p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
            
            dist_table = Table(title="📈 延迟分布", style="blue")
            dist_table.add_column("百分位")
            dist_table.add_column("延迟")
            dist_table.add_row("90%", f"{p90:.3f}s")
            dist_table.add_row("95%", f"{p95:.3f}s")
            dist_table.add_row("99%", f"{p99:.3f}s")
            console.print(dist_table)
    
    # 打印示例响应
    console.print("\n🔍 示例响应:", style="bold")
    sample_count = min(3, len(results))
    for res in results[:sample_count]:
        if "response" in res:
            console.print(f"[bold green][请求 {res['client_id']}][/bold green] vLLM ID: {res.get('vllm_id', 'N/A')}")
            console.print(f"  耗时: [cyan]{res['latency']:.3f}s[/cyan] | Tokens: [cyan]{res.get('tokens_used', 0)}[/cyan]")
            console.print(f"  问题: {res['prompt'][:80]}{'...' if len(res['prompt']) > 80 else ''}")
            console.print(f"  回答: {res['response'][:120]}{'...' if len(res['response']) > 120 else ''}\n")
        elif "error" in res:
            console.print(f"[bold red][请求 {res['client_id']}][/bold red] 错误: {res['error'][:120]}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="vLLM 批量请求生成器")
    
    # 基本参数
    parser.add_argument("--num-requests", type=int, default=10, help="请求总数")
    parser.add_argument("--max-concurrent", type=int, default=5, help="最大并发请求数")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="API端点")
    parser.add_argument("--prompt-file", type=str, help="包含提示词的文件路径（每行一个）")
    parser.add_argument("--output-file", type=str, default="vllm_results.json", help="输出结果文件名")
    
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
    
    # 准备提示词
    if args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
        if not prompts:
            print("⚠️ 使用默认提示词")
            prompts = [f"测试提示 #{i}" for i in range(args.num_requests)]
        else:
            # 如果文件中的提示词少于请求数，循环使用
            if len(prompts) < args.num_requests:
                prompts = (prompts * (args.num_requests // len(prompts) + 1))[:args.num_requests]
            else:
                prompts = prompts[:args.num_requests]
    else:
        prompts = [f"测试提示 #{i}" for i in range(args.num_requests)]
    
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
    
    print(f"🚀 启动 vLLM 压力测试")
    print(f"├─ 模型: {args.model}")
    print(f"├─ 端点: {args.endpoint}")
    print(f"├─ 请求数: {args.num_requests}")
    print(f"├─ 并发数: {args.max_concurrent}")
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
    
    # 使用Rich进度条
    try:
        from rich.progress import Progress
    except ImportError:
        print("⚠️ 未找到rich库，使用简单进度指示")
        progress = None
    else:
        progress = Progress()
    
    # 运行测试
    start_time = time.time()
    if progress:
        with progress:
            results = asyncio.run(
                batch_request_vllm(
                    prompts=prompts,
                    max_concurrent=args.max_concurrent,
                    model=args.model,
                    endpoint=args.endpoint,
                    progress=progress,
                    **request_params
                )
            )
    else:
        # 没有rich时的简单实现
        results = asyncio.run(
            batch_request_vllm(
                prompts=prompts,
                max_concurrent=args.max_concurrent,
                model=args.model,
                endpoint=args.endpoint,
                progress=None,
                **request_params
            )
        )
        print(f"已完成 {len(prompts)}/{len(prompts)} 请求")
    
    total_time = time.time() - start_time
    
    # 打印统计信息
    print_statistics(results, total_time)
    
    # 保存结果
    save_results_to_file(results, args.output_file)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n操作被用户中断")