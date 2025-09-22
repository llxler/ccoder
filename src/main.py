import os
import json
import time 
import signal 
from generator import CGenerator
from utils import DS_REPO_DIR, DS_FILE, DS_GRAPH_DIR, PT_FILE, MODEL
from argparse import ArgumentParser


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("处理超时")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', default=MODEL, help='代码模型，支持: deepseekcoder, codegen, codegen25, santacoder, starcoder, codellama, gpt35, gpt4')
    parser.add_argument('-f', '--file', default=PT_FILE, help='输出提示文件路径')
    parser.add_argument('-c', '--c_dataset', default=None, help='C语言数据集文件路径，不指定则使用默认路径')
    parser.add_argument('-t', '--timeout', type=int, default=30, help='单个样本处理超时时间（秒）')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='批处理大小，每处理这么多样本保存一次结果')
    args = parser.parse_args()
    print(f'使用模型: {args.model}')
    print(f'输出提示文件: {args.file}')
    print(f'C语言数据集文件: {args.c_dataset}')
    print(f'单个样本处理超时时间: {args.timeout}秒')
    print(f'批处理大小: {args.batch_size}')
    generator = CGenerator(DS_REPO_DIR, DS_GRAPH_DIR, args.model.lower())

    dataset_file = args.c_dataset if args.c_dataset else DS_FILE
    with open(dataset_file, 'r') as f:
        dataset = [json.loads(line) for line in f.readlines()]
    print(f'总共有 {len(dataset)} 个样本待处理')
    
    start_idx = 0
    if os.path.exists(args.file):
        with open(args.file, 'r') as f:
            processed = len(f.readlines())
            if processed > 0:
                start_idx = processed
                print(f'检测到{args.file}已处理 {processed} 个样本，从第 {start_idx+1} 个样本继续处理')
                
    ret = []
    timeout_samples = []  
    
    signal.signal(signal.SIGALRM, timeout_handler)
    
    for i, item in enumerate(dataset[start_idx:], start=start_idx):
        if i % 10 == 0:
            print(f'正在处理第 {i}/{len(dataset)} 个样本...')
            
        if i > start_idx and (i - start_idx) % args.batch_size == 0:
            print(f'正在保存批处理结果... 已完成 {i} 个样本')
            with open(args.file, 'a', encoding="utf-8") as f:
                for result_item in ret:
                    json.dump(result_item, f, ensure_ascii=False)
                    f.write('\n')
            ret = []  
            
        fpath = os.path.join(DS_REPO_DIR, item['fpath'])
        try:
            if fpath.endswith('.c') or fpath.endswith('.h'):
                signal.alarm(args.timeout)
                
                start_time = time.time()
                prompt_text = generator.retrieve_prompt(item['pkg'], fpath, item['input'])
                
                signal.alarm(0)
                
                process_time = time.time() - start_time
                if process_time > 5: 
                    print(f'样本 {i} 处理时间较长: {process_time:.2f}秒')
                
                result = {
                    "id": item.get('id', i+1),  
                    "prompt": prompt_text
                }
                ret.append(result)
            else:
                print(f'跳过非C语言文件: {fpath}')
        except TimeoutException:
            print(f'警告: 处理样本 {i}, 文件 {item["fpath"]} 超时，已跳过')
            timeout_samples.append({"id": item.get('id', i+1), "fpath": item["fpath"]})
            signal.alarm(0)
            continue
        except Exception as e:
            print(f'处理样本 {i}, 文件 {item["fpath"]} 时出错')
            print(repr(e))
            signal.alarm(0)
            continue

    print(f'成功为 {len(ret) + (i - start_idx + 1 - len(ret) - len(timeout_samples))} 个样本生成提示')
    print(f'跳过了 {len(timeout_samples)} 个超时样本')
    
    if ret:
        with open(args.file, 'a', encoding="utf-8") as f:
            for item in ret:
                json.dump(item, f)
                f.write('\n')
    
    if timeout_samples:
        timeout_file = args.file + '.timeout'
        with open(timeout_file, 'w') as f:
            for item in timeout_samples:
                json.dump(item, f)
                f.write('\n')
        print(f'超时样本信息已保存到 {timeout_file}')
    
    print(f'结果已保存到 {args.file}') 