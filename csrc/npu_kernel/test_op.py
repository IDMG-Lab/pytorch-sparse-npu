import time
import statistics

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib

torch.npu.config.allow_internal_format = False

import numpy as np
import sys
from torch_scatter import gather_csr
import torch_sparse
import torch_cluster

# 为了让 case3 的随机部分可复现
torch.manual_seed(0)

# =========================
# Test data
# =========================
tests_ind2ptr = {
    # case1：原 2x8 -> flatten 成 16，再全局排序（显式给出结果）
    'case1': {
        'ind': torch.tensor(
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4],
            dtype=torch.int64
        ),
        'M': 5
    },

    # case2：8*1024=8192 个元素；每个值 0..255 各出现 32 次，本身就是全局有序
    'case2': {
        'ind': torch.repeat_interleave(
            torch.arange(256, dtype=torch.int64), 32
        ),  # shape: [8192]
        'M': 256
    },

    # case3：先构造，再全局排序（shape: [8192]，整体非降序）
    'case3': {
        'ind': torch.sort(torch.cat([
            torch.zeros(8 * 400, dtype=torch.int64),                 # 大量 0
            torch.full((8 * 400,), 255, dtype=torch.int64),          # 大量 255
            torch.randint(0, 256, (8 * 224,), dtype=torch.int64),    # 剩余随机（可复现）
        ], dim=0)).values,
        'M': 256
    },
}

case_data = {
    'case1': {
        'ind': torch.randint(0, 256, (8, 1024), dtype=torch.int64),  # ind范围是[0,256)
        'M': 256
    },
    "case4": {
        "input": torch.randn(2, 8, 1, 1, dtype=torch.bfloat16),   # shape: [2, 8, 1, 1]
        "other": torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)  # shape: [1, 1, 32, 64]
    },

    # -------- large tensor --------
    "case_large": {
        "input": np.random.randn(1, 1, 310, 10000).astype(np.int64),
        "other": np.random.randn(1, 1, 310, 10000).astype(np.int64)
    }
}

tests_radius = {
    # -------- radius_graph small / easy to verify --------
    "case1": {
        "pos": torch.tensor([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.0, 0.2],
            [1.0, 1.0],
            [1.1, 1.0],
            [2.0, 2.0],
            [2.05, 2.0],
            [3.0, 3.0],
            [3.2, 3.2],
        ], dtype=torch.float32),              # shape: [10, 2]
        "batch": None,                        # 单图可以传 None
        "r": 0.25,
        "max_num_neighbors": 32,
        "loop": False,
        "ptr": torch.tensor([0, 10], dtype=torch.int32),
    },

    # -------- radius_graph with batch (multiple graphs) --------
    "case2": {
        "pos": torch.randn(40, 3, dtype=torch.float32),  # shape: [40, 3]
        "batch": torch.tensor([0] * 16 + [1] * 24, dtype=torch.int64),  # shape: [40]
        "r": 0.6,
        "max_num_neighbors": 16,
        "loop": False,
        "ptr": torch.tensor([0, 16, 40], dtype=torch.int32),
    },

    # -------- larger stress test --------
    "case3": {
        "pos": torch.randn(5000, 3, dtype=torch.float32),  # shape: [5000, 3]
        "batch": torch.zeros(5000, dtype=torch.int64),     # 单图但显式 batch
        "r": 0.2,
        "max_num_neighbors": 32,
        "loop": False,
        "ptr": torch.tensor([0, 5000], dtype=torch.int32),
    },
}

# =========================
# Utils
# =========================
def verify_result(real_result, golden):
    if golden.dtype == torch.float32:
        rtol = 1e-4
        atol = 1e-4
    else:
        rtol = 1e-3
        atol = 1e-3

    minimum = 10e-10
    golden = torch.where(golden == 0, minimum, golden)
    real_result = torch.where(real_result == 0, minimum, real_result)

    abs_diff = torch.abs(real_result - golden)
    rel_diff = abs_diff / torch.max(torch.abs(real_result), torch.abs(golden))
    is_close = (abs_diff <= atol) | (rel_diff <= rtol)
    both_nan = torch.isnan(real_result) & torch.isnan(golden)
    is_close = is_close | both_nan
    err_num = torch.sum(~is_close).item()

    if real_result.numel() * rtol < err_num:
        print(f"[ERROR] result error")
        return False
    print("test pass")
    return True


def bench_per_iter_sync(fn, warmup=20, iters=200, sync=None):
    """
    每次迭代都 sync：测“延迟”(包含 launch+同步等固定开销)。
    """
    for _ in range(warmup):
        fn()
    if sync:
        sync()

    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "mean_ms": 1000.0 * sum(times) / len(times),
        "median_ms": 1000.0 * statistics.median(times),
        "min_ms": 1000.0 * min(times),
        "max_ms": 1000.0 * max(times),
    }


def bench_total_one_sync(fn, warmup=50, iters=10000, sync=None):
    """
    只在整体前后 sync 一次：测“吞吐”(减少每次同步带来的放大)。
    返回 ms/iter。
    """
    for _ in range(warmup):
        fn()
    if sync:
        sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync:
        sync()
    t1 = time.perf_counter()
    return 1000.0 * (t1 - t0) / iters


def bench_npu_event(fn, warmup=50, iters=5000):
    """
    尝试用 NPU Event 测设备侧时间（更接近 kernel 时间）。
    如果环境不支持，会抛异常，外面 catch。
    返回 ms/iter。
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / iters


# =========================
# Your original test classes (mostly kept)
# =========================
class TestCustomOP(TestCase):
    def test_op(self, num):
        print(num)
        caseName = 'case' + str(num)
        input_x = None
        input_other = None
        if int(num) == 3 or int(num) == 4:
            input_x = case_data[caseName]["input"]
            input_other = case_data[caseName]["other"]
        else:
            input_x = torch.from_numpy(case_data[num]["input"])
            input_other = torch.from_numpy(case_data[num]["other"])

        output = torch.fmax(input_x, input_other)

        output_npu = custom_ops_lib.custom_op(input_x.npu(), input_other.npu())
        if output_npu is None:
            print(f"{caseName} execution timed out!")
        else:
            if verify_result(output_npu.cpu(), output):
                print(f"{caseName} verify result pass!")
            else:
                print(f"{caseName} verify result failed!")


class Test_Gathercsr(TestCase):
    def generate_input(self, n):
        rows_non_zero_elements = np.random.randint(0, 100, size=n)
        indptr = [0]
        for i in range(n):
            indptr.append(indptr[-1] + rows_non_zero_elements[i])
        indptr_tensor = torch.tensor(indptr, dtype=torch.int64)
        src = torch.arange(10, 10 * n + 1, 10, dtype=torch.int64)
        print(f"indptr_tensor:{indptr_tensor}\n")
        return src, indptr_tensor

    def test_op(self):
        for n in range(10, 20, 10):
            src, ptr = self.generate_input(n)
            output_npu = custom_ops_lib.Gathercsr(src.npu(), ptr.npu())
            output = gather_csr(src, ptr)
            print('------------------------------Gathercsr算子运行成功-------------------------')
            print(output)
            print(output_npu.cpu())
            if output_npu is None:
                print(f"execution timed out!")
            else:
                if verify_result(output_npu.cpu(), output):
                    print(f"verify result pass!")
                else:
                    print(f"verify result failed!")


# =========================
# ✅ Ind2ptr: correctness + 3种计时方式 + 可选大case + 可选int32
# =========================
class Test_Ind2ptr(TestCase):
    def test_op(self):
        # 你可以按需开关
        ENABLE_E2E = True          # 是否测端到端（包含H2D）
        ENABLE_EVENT = False        # 是否尝试 Event 计时
        ENABLE_INT32 = False        # 是否尝试 int32 版本（如果NPU实现支持）
        ENABLE_BIG_CASE = False     # 是否加一个更大规模case看吞吐
        BIG_N = 1_000_000          # 大case ind长度（可调大/小）

        # 可选：让CPU更稳定一些（线程数会影响结果）
        # torch.set_num_threads(1)

        # --------- 先跑原case1~case3 ---------
        for i in range(1, 4):
            case_name = f'case{i}'
            input_ind = tests_ind2ptr[case_name]["ind"]
            input_M = tests_ind2ptr[case_name]["M"]

            # correctness
            input_ind_npu = input_ind.npu()
            output_npu = custom_ops_lib.Ind2ptr(input_ind_npu, input_M)
            output_cpu = torch.ops.torch_sparse.ind2ptr(input_ind, input_M)

            print(f"\n================ {case_name} ================")
            if output_npu is None:
                print(f"{case_name} execution timed out!")
                continue

            ok = verify_result(output_npu.cpu(), output_cpu)
            print(f"{case_name} verify: {'PASS' if ok else 'FAIL'}")
            if not ok:
                continue

            # --------- 计时口径 1：每次迭代都同步（延迟 / launch开销放大）---------
            cpu_lat = bench_per_iter_sync(
                fn=lambda: torch.ops.torch_sparse.ind2ptr(input_ind, input_M),
                warmup=50, iters=500, sync=None
            )
            npu_lat = bench_per_iter_sync(
                fn=lambda: custom_ops_lib.Ind2ptr(input_ind_npu, input_M),
                warmup=100, iters=800, sync=torch.npu.synchronize
            )
            print("[Latency][CPU] mean={mean_ms:.4f} ms, median={median_ms:.4f} ms, min={min_ms:.4f} ms, max={max_ms:.4f} ms".format(**cpu_lat))
            print("[Latency][NPU] mean={mean_ms:.4f} ms, median={median_ms:.4f} ms, min={min_ms:.4f} ms, max={max_ms:.4f} ms".format(**npu_lat))

            # --------- 计时口径 2：整体只同步一次（吞吐 / 更公平）---------
            cpu_tp = bench_total_one_sync(
                fn=lambda: torch.ops.torch_sparse.ind2ptr(input_ind, input_M),
                warmup=50, iters=20000, sync=None
            )
            npu_tp = bench_total_one_sync(
                fn=lambda: custom_ops_lib.Ind2ptr(input_ind_npu, input_M),
                warmup=200, iters=20000, sync=torch.npu.synchronize
            )
            print(f"[Throughput][CPU] {cpu_tp:.6f} ms/iter  (整体前后sync一次)")
            print(f"[Throughput][NPU] {npu_tp:.6f} ms/iter  (整体前后sync一次)")

            # --------- 计时口径 3：NPU Event（更接近设备侧时间）---------
            if ENABLE_EVENT:
                try:
                    npu_evt = bench_npu_event(
                        fn=lambda: custom_ops_lib.Ind2ptr(input_ind_npu, input_M),
                        warmup=200, iters=50000
                    )
                    print(f"[Event][NPU] {npu_evt:.6f} ms/iter  (NPU Event)")
                except Exception as e:
                    print(f"[Event][NPU] not supported or failed: {repr(e)}")

            # --------- 可选：端到端（包含H2D拷贝）---------
            if ENABLE_E2E:
                e2e = bench_total_one_sync(
                    fn=lambda: custom_ops_lib.Ind2ptr(input_ind.npu(), input_M),
                    warmup=50, iters=5000, sync=torch.npu.synchronize
                )
                print(f"[E2E][NPU] {e2e:.6f} ms/iter  (含 H2D: input_ind.npu())")

        # --------- 加一个更大规模的 case，看 NPU 吞吐是否会更接近/超过 CPU ---------
        if ENABLE_BIG_CASE:
            print(f"\n================ case_big (N={BIG_N}) ================")
            big_ind = torch.sort(torch.randint(0, 256, (BIG_N,), dtype=torch.int64)).values
            big_M = 256
            big_ind_npu = big_ind.npu()

            # correctness（只做一次）
            out_big_npu = custom_ops_lib.Ind2ptr(big_ind_npu, big_M)
            out_big_cpu = torch.ops.torch_sparse.ind2ptr(big_ind, big_M)
            ok_big = verify_result(out_big_npu.cpu(), out_big_cpu)
            print(f"case_big verify: {'PASS' if ok_big else 'FAIL'}")
            if ok_big:
                cpu_tp_big = bench_total_one_sync(
                    fn=lambda: torch.ops.torch_sparse.ind2ptr(big_ind, big_M),
                    warmup=10, iters=200, sync=None
                )
                npu_tp_big = bench_total_one_sync(
                    fn=lambda: custom_ops_lib.Ind2ptr(big_ind_npu, big_M),
                    warmup=50, iters=2000, sync=torch.npu.synchronize
                )
                print(f"[Throughput][CPU][big] {cpu_tp_big:.6f} ms/iter")
                print(f"[Throughput][NPU][big] {npu_tp_big:.6f} ms/iter")

                if ENABLE_EVENT:
                    try:
                        npu_evt_big = bench_npu_event(
                            fn=lambda: custom_ops_lib.Ind2ptr(big_ind_npu, big_M),
                            warmup=50, iters=5000
                        )
                        print(f"[Event][NPU][big] {npu_evt_big:.6f} ms/iter")
                    except Exception as e:
                        print(f"[Event][NPU][big] not supported or failed: {repr(e)}")


class Test_Radius(TestCase):
    def test_op(self):
        for i in range(1, 4):
            casename = 'case' + str(i)
            d = tests_radius[casename]
            x = d["pos"]
            r = d["r"]
            batch = d["batch"]
            loop = d["loop"]
            ptr = d["ptr"]
            max_num_neighbors = d["max_num_neighbors"]

            edge_index_npu = custom_ops_lib.Radius(x.npu(), x.npu(), ptr.npu(), ptr.npu(), r, max_num_neighbors, False)
            edge_index = torch.ops.torch_cluster.radius(x, x, ptr.long(), ptr.long(), r, max_num_neighbors, 1)

            flow = 'source_to_target'
            if flow == 'source_to_target':
                row, col = edge_index[1], edge_index[0]
                edge_index = edge_index.flip(0)
            else:
                row, col = edge_index[0], edge_index[1]

            print('------------------------------Radius算子运行成功-------------------------')
            print(edge_index)
            print(edge_index_npu.cpu())
            if edge_index_npu is None:
                print(f"{casename} execution timed out!")
            else:
                if verify_result(edge_index_npu.cpu(), edge_index):
                    print(f"{casename} verify result pass!")
                else:
                    print(f"{casename} verify result failed!")


if __name__ == "__main__":
    Test_Ind2ptr().test_op()
    print("---------------------------------------\n可以运行多个算子啦！！！！---------------------------------------\n")
