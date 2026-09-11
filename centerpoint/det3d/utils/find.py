"""环境探测工具。

用于查找本机的 Anaconda、CUDA 安装路径，检测 GPU 计算能力（sm_xx）并查询
GPU 显存占用情况，供构建脚本与运行时使用。
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import fire


def _get_info_from_anaconda_info(info, split=":"):
    """解析 `conda info` 输出为 dict（多行值会转成列表）。"""
    info = info.strip("\n").replace(" ", "")
    info_dict = {}
    latest_key = ""
    for line in info.splitlines():
        if split in line:
            pair = line.split(split)
            info_dict[pair[0]] = pair[1]
            latest_key = pair[0]
        else:
            if not isinstance(info_dict[latest_key], list):
                info_dict[latest_key] = [info_dict[latest_key]]
            info_dict[latest_key].append(line)
    return info_dict


def find_anaconda():
    """查找 Anaconda 安装路径。"""
    # 先尝试默认路径
    path = Path.home() / "anaconda3"
    if path.exists():
        return path
    # 再尝试命令行 conda info
    try:
        info = subprocess.check_output("conda info", shell=True).decode("utf-8")
        info_dict = _get_info_from_anaconda_info(info)
        return info_dict["activeenvlocation"]
    except subprocess.CalledProcessError:
        raise RuntimeError("find anadonda failed")


def find_cuda():
    """查找 CUDA 安装路径。"""
    # 猜测一：从环境变量读取
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # 猜测二：按平台默认安装路径
        if sys.platform == "win32":
            cuda_homes = glob.glob(
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*"
            )
            if len(cuda_homes) == 0:
                cuda_home = ""
            else:
                cuda_home = cuda_homes[0]
        else:
            cuda_home = "/usr/local/cuda"
        if not os.path.exists(cuda_home):
            # 猜测三：通过 nvcc 可执行文件定位
            try:
                which = "where" if sys.platform == "win32" else "which"
                nvcc = subprocess.check_output([which, "nvcc"]).decode().rstrip("\r\n")
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
            except Exception:
                cuda_home = None
    if cuda_home is None:
        raise RuntimeError(
            "No CUDA runtime is found, using CUDA_HOME='{}'".format(cuda_home)
        )
    return cuda_home


def find_cuda_device_arch():
    """检测当前 GPU 的 CUDA 计算能力。

    先尝试使用 CUDA 自带的 deviceQuery 工具；失败则编译一段临时 CUDA 程序查询，
    并根据 nvcc 支持的架构版本回退降级，返回如 "sm_75" 的字符串，失败返回 None。
    """
    if sys.platform == "win32":
        # TODO: 暂不支持 Windows
        return None
    cuda_home = find_cuda()
    if cuda_home is None:
        return None
    cuda_home = Path(cuda_home)
    try:
        device_query_path = cuda_home / "extras/demo_suite/deviceQuery"
        if not device_query_path.exists():
            # 编译临时程序查询设备计算能力。
            source = """
            #include <cuda_runtime.h>
            #include <iostream>
            int main(){
                int nDevices;
                cudaGetDeviceCount(&nDevices);
                for (int i = 0; i < nDevices; i++) {
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, i);
                    std::cout << prop.major << "." << prop.minor << std::endl;
                }
                return 0;
            }
            """
            with tempfile.NamedTemporaryFile("w", suffix=".cc") as f:
                f_path = Path(f.name)
                f.write(source)
                f.flush()
                try:
                    # TODO: 暂不支持 Windows
                    cmd = (
                        f"g++ {f.name} -o {f_path.stem}"
                        f" -I{cuda_home / 'include'} -L{cuda_home / 'lib64'} -lcudart"
                    )
                    print(cmd)
                    subprocess.check_output(cmd, shell=True, cwd=f_path.parent)
                    cmd = f"./{f_path.stem}"
                    arches = (
                        subprocess.check_output(cmd, shell=True, cwd=f_path.parent)
                        .decode()
                        .rstrip("\r\n")
                        .split("\n")
                    )
                    if len(arches) < 1:
                        return None
                    arch = arches[0]
                except Exception:
                    return None
        else:
            cmd = f"{str(device_query_path)} | grep 'CUDA Capability'"
            arch = (
                subprocess.check_output(cmd, shell=True)
                .decode()
                .rstrip("\r\n")
                .split(" ")[-1]
            )
        # assert len(arch) == 2
        arch_list = [int(s) for s in arch.split(".")]
        arch_int = arch_list[0] * 10 + arch_list[1]
        find_work_arch = False
        # 从当前架构向下回退，找到 nvcc 支持的架构。
        while arch_int > 10:
            try:
                res = subprocess.check_output(
                    "nvcc -arch=sm_{}".format(arch_int),
                    shell=True,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as e:
                if "No input files specified" in e.output.decode():
                    find_work_arch = True
                    break
                elif (
                    "is not defined for option 'gpu-architecture'" in e.output.decode()
                ):
                    arch_int -= 1
                else:
                    raise RuntimeError("unknown error")
        if find_work_arch:
            arch = f"sm_{arch_int}"
        else:
            arch = None

    except Exception:
        arch = None
    return arch


def get_gpu_memory_usage():
    """查询各 GPU 的空闲/总显存（字节），以 JSON 数组返回；失败返回 None。"""
    if sys.platform == "win32":
        # TODO: 暂不支持 Windows
        return None
    cuda_home = find_cuda()
    if cuda_home is None:
        return None
    cuda_home = Path(cuda_home)
    # 编译临时 CUDA 程序查询显存。
    source = """
    #include <cuda_runtime.h>
    #include <iostream>
    int main(){
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        size_t free_m, total_m;
        // 输出 json 格式。
        std::cout << "[";
        for (int i = 0; i < nDevices; i++) {
            cudaSetDevice(i);
            cudaMemGetInfo(&free_m, &total_m);
            std::cout << "[" << free_m << "," << total_m << "]";
            if (i != nDevices - 1)
                std::cout << "," << std::endl;
        }
        std::cout << "]" << std::endl;
        return 0;
    }
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cc") as f:
        f_path = Path(f.name)
        f.write(source)
        f.flush()
        try:
            # TODO: 暂不支持 Windows
            cmd = (
                f"g++ {f.name} -o {f_path.stem} -std=c++11"
                f" -I{cuda_home / 'include'} -L{cuda_home / 'lib64'} -lcudart"
            )
            print(cmd)
            subprocess.check_output(cmd, shell=True, cwd=f_path.parent)
            cmd = f"./{f_path.stem}"
            usages = subprocess.check_output(
                cmd, shell=True, cwd=f_path.parent
            ).decode()
            usages = json.loads(usages)
            return usages
        except Exception:
            return None
    return None


if __name__ == "__main__":
    print(find_cuda_device_arch())
    # fire.Fire()
