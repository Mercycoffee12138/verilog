import subprocess
import re
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional

# --- 配置 ---
CPP_SOURCE_FILE = "lesson1_sourcefile.cpp"
EXECUTABLE_NAME = "outputfile"  # 在 Windows 上可能是 "outputfile.exe"
# 如果 mpirun 不是直接可用，或者需要指定路径，请修改下面的命令
# 例如: "C:\\Program Files\\Microsoft MPI\\Bin\\mpiexec.exe"
MPI_RUN_CMD = "mpirun"
MPI_PROCESSES = 4  # MPI 进程数

# 定义要测试的 N, M, P 组合
# 你可以根据之前的建议添加更多组合
CONFIGURATIONS: List[Tuple[int, int, int]] = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 64, 2048),    # 一个较小的 M
    (1024, 2048, 512),   # M > N, P
    (4096, 64, 4096),    # 当前 C++ 代码中的一个例子
    # (2048, 2048, 2048), # 如果需要更大的测试
]

# 确保在 Windows 上使用正确的 .exe 后缀 (如果编译器生成的话)
if os.name == 'nt' and not EXECUTABLE_NAME.endswith(".exe"):
    EXECUTABLE_NAME_WITH_EXT = EXECUTABLE_NAME + ".exe"
else:
    EXECUTABLE_NAME_WITH_EXT = EXECUTABLE_NAME

def modify_cpp_source(n: int, m: int, p: int) -> bool:
    """修改 C++ 源文件中的 N, M, P 值"""
    try:
        with open(CPP_SOURCE_FILE, 'r', encoding='utf-8') as f:
            content = f.readlines()

        new_content = []
        modified = False
        # 寻找类似 "const int N = 4096, M = 64, P = 4096;" 的行
        # 正则表达式会更精确，但简单的字符串查找和替换对于固定格式也可能有效
        # 使用更健壮的正则表达式
        nmp_line_pattern = re.compile(r"(\s*const\s+int\s+N\s*=\s*)(\d+)(\s*,\s*M\s*=\s*)(\d+)(\s*,\s*P\s*=\s*)(\d+)(\s*;)")

        for line in content:
            match = nmp_line_pattern.search(line)
            if match:
                new_line = f"{match.group(1)}{n}{match.group(3)}{m}{match.group(5)}{p}{match.group(7)}\n"
                new_content.append(new_line)
                modified = True
                print(f"  Source modified: N={n}, M={m}, P={p}")
            else:
                new_content.append(line)
        
        if not modified:
            print(f"  ERROR: Could not find the N, M, P definition line in {CPP_SOURCE_FILE}")
            return False

        with open(CPP_SOURCE_FILE, 'w', encoding='utf-8') as f:
            f.writelines(new_content)
        return True
    except Exception as e:
        print(f"  ERROR modifying C++ source: {e}")
        return False

def compile_cpp() -> bool:
    """编译 C++ 代码"""
    compile_command = ["mpic++", "-fopenmp", "-o", EXECUTABLE_NAME, CPP_SOURCE_FILE]
    if os.name == 'nt': # MinGW g++ on Windows might not need -fopenmp with mpicc wrapper, adjust if needed
        pass # Assuming mpic++ handles OpenMP flags correctly or it's linked by default

    print(f"  Compiling: {' '.join(compile_command)}")
    try:
        result = subprocess.run(compile_command, capture_output=True, text=True, check=True, encoding='utf-8')
        if result.stderr:
            print(f"  Compilation warnings:\n{result.stderr}")
        print("  Compilation successful.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Compilation failed.")
        print(f"  Command: {e.cmd}")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout:\n{e.stdout}")
        print(f"  Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"  ERROR: mpic++ command not found. Is MPI installed and in PATH?")
        return False


def run_and_parse_output(n: int, m: int, p: int) -> List[Dict]:
    """运行 C++ 可执行文件并解析其输出"""
    run_command = [MPI_RUN_CMD, "-np", str(MPI_PROCESSES), EXECUTABLE_NAME_WITH_EXT]
    
    results_for_config = []
    print(f"  Running: {' '.join(run_command)}")
    try:
        # 增加超时以防程序挂起
        process = subprocess.run(run_command, capture_output=True, text=True, check=True, encoding='utf-8', timeout=600) # 10分钟超时
        output = process.stdout
        # print(f"Raw output from C++:\n{output[:1000]}...") # 打印部分原始输出用于调试

        # 解析算法名称、时间和 GFLOPS
        # 例如: [Baseline] Time: 123.45 ms
        #       [Baseline] GFLOPS: 67.89
        time_pattern = re.compile(r"\[(.*?)\] Time: ([\d.]+) ms")
        gflops_pattern = re.compile(r"\[(.*?)\] GFLOPS: ([\d.]+)")

        # 使用字典临时存储，因为 GFLOPS 可能在 Time 之后出现
        algo_data = {}

        for line in output.splitlines():
            time_match = time_pattern.search(line)
            if time_match:
                algo_name = time_match.group(1).strip()
                time_val = float(time_match.group(2))
                if algo_name not in algo_data:
                    algo_data[algo_name] = {}
                algo_data[algo_name]['Time (ms)'] = time_val

            gflops_match = gflops_pattern.search(line)
            if gflops_match:
                algo_name = gflops_match.group(1).strip()
                gflops_val = float(gflops_match.group(2))
                if algo_name not in algo_data:
                    algo_data[algo_name] = {}
                algo_data[algo_name]['GFLOPS'] = gflops_val
        
        for algo_name, data in algo_data.items():
            results_for_config.append({
                "N": n, "M": m, "P": p,
                "Algorithm": algo_name,
                "Time (ms)": data.get('Time (ms)'),
                "GFLOPS": data.get('GFLOPS')
            })
        
        if not results_for_config:
            print("  WARNING: No performance data parsed from output.")
            print("  First 10 lines of output:")
            for i, line in enumerate(output.splitlines()):
                if i < 10: print(f"    {line}")
                else: break


    except subprocess.CalledProcessError as e:
        print(f"  ERROR: C++ program execution failed for N={n},M={m},P={p}.")
        print(f"  Command: {e.cmd}")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout:\n{e.stdout}")
        print(f"  Stderr:\n{e.stderr}")
    except subprocess.TimeoutExpired:
        print(f"  ERROR: C++ program timed out for N={n},M={m},P={p}.")
    except FileNotFoundError:
        print(f"  ERROR: {MPI_RUN_CMD} or {EXECUTABLE_NAME_WITH_EXT} not found.")
    except Exception as e:
        print(f"  An unexpected error occurred during execution or parsing: {e}")
        
    return results_for_config

def main():
    all_results_data = []
    original_nmp = None # 用于恢复原始 N,M,P

    # 尝试读取原始 N,M,P 以便后续恢复
    try:
        with open(CPP_SOURCE_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        nmp_line_pattern = re.compile(r"const\s+int\s+N\s*=\s*(\d+)\s*,\s*M\s*=\s*(\d+)\s*,\s*P\s*=\s*(\d+)\s*;")
        match = nmp_line_pattern.search(content)
        if match:
            original_nmp = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            print(f"Original N,M,P found: {original_nmp}")
    except Exception as e:
        print(f"Could not read original N,M,P: {e}")


    for n, m, p in CONFIGURATIONS:
        print(f"\nProcessing configuration: N={n}, M={m}, P={p}")

        if not modify_cpp_source(n, m, p):
            print("  Skipping this configuration due to source modification error.")
            continue

        if not compile_cpp():
            print("  Skipping this configuration due to compilation error.")
            continue
        
        # 清理旧的可执行文件，以防万一 (主要针对 Windows)
        if os.name == 'nt' and os.path.exists(EXECUTABLE_NAME):
             if EXECUTABLE_NAME_WITH_EXT != EXECUTABLE_NAME and os.path.exists(EXECUTABLE_NAME):
                try:
                    os.remove(EXECUTABLE_NAME)
                except OSError as e:
                    print(f"  Warning: Could not remove old executable {EXECUTABLE_NAME}: {e}")


        results = run_and_parse_output(n, m, p)
        all_results_data.extend(results)

    # 恢复原始的 N, M, P 值
    if original_nmp:
        print(f"\nRestoring original N,M,P values in {CPP_SOURCE_FILE} to {original_nmp}...")
        if modify_cpp_source(original_nmp[0], original_nmp[1], original_nmp[2]):
            print("  Source file restored.")
            # Optionally recompile to the original state
            # compile_cpp()
        else:
            print("  ERROR: Failed to restore original N,M,P values.")
    else:
        print(f"\nWarning: Original N,M,P values not found, {CPP_SOURCE_FILE} might be left in the last tested state.")


    if all_results_data:
        df = pd.DataFrame(all_results_data)
        print("\n\n--- Performance Comparison ---")
        
        # 根据 N, M, P 和算法排序，使表格更易读
        df_sorted = df.sort_values(by=["N", "M", "P", "Algorithm"])
        
        # 使用 to_string() 打印整个 DataFrame
        print(df_sorted.to_string(index=False))

        # 你也可以将结果保存到 CSV 文件
        # csv_filename = "matmul_performance_results.csv"
        # df_sorted.to_csv(csv_filename, index=False)
        # print(f"\nResults also saved to {csv_filename}")
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    main()