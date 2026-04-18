from .schedule_pool_no_threading import SchedulePool
import cost_model_config


import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict


TOTAL_CPU = os.cpu_count() or 64
OUTER_WORKERS = 1          # 同时跑几个 iteration
NGPUS = 16
SCHEDULE_TYPE = "genetic"
MACHINE_SIZE = 8

# 每个 iteration 里，GA 最多能用多少核
CORES_PER_ITER = max(1, (TOTAL_CPU - 2) // OUTER_WORKERS)


# =========================
# 🔒 每个 iteration 的“独立进程入口”
# =========================
def run_one_iteration(
    iteration: int,
    pool_kwargs: Dict,
    cores_per_iter: int,
):
    """
    每个 iteration 在【独立进程】中运行
    """

    # 1️⃣ 为 joblib 设置独立的临时目录（非常重要）
    joblib_tmp = f"/tmp/joblib_iter_{iteration}_pid_{os.getpid()}"
    os.makedirs(joblib_tmp, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = joblib_tmp

    # 2️⃣ 禁止 BLAS / OpenMP 再偷偷开线程（防止 CPU 超卖）
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # 3️⃣ 可选：把 GA 的并行核数通过环境变量传进去
    os.environ["GA_NUM_CORES"] = str(cores_per_iter)

    # 4️⃣ 真正跑 schedule
    pool = SchedulePool(**pool_kwargs)
    pool.submit(iteration)

    path = os.path.join(pool.output_dir, f"schedule_{iteration}.yaml")
    return path


if __name__ == "__main__":

    # ✅ 非常重要：外层强制 spawn
    mp.set_start_method("spawn", force=True)

    iterations = list(range(50))  # 你要并行跑的 iteration

    pool_kwargs = dict(
        num_workers=1,
        n_gpus=NGPUS,
        machine_size=MACHINE_SIZE,
        schedule_type=SCHEDULE_TYPE,
    )

    print(
        f"🚀 Launching {len(iterations)} iterations | "
        f"outer={OUTER_WORKERS}, inner(joblib)={CORES_PER_ITER}"
    )

    with ProcessPoolExecutor(
        max_workers=OUTER_WORKERS,
        mp_context=mp.get_context("spawn"),
    ) as executor:

        futures = [
            executor.submit(
                run_one_iteration,
                it,
                pool_kwargs,
                CORES_PER_ITER,
            )
            for it in iterations
        ]

        for f in as_completed(futures):
            try:
                print("✅ Finished:", f.result())
            except Exception as e:
                print("❌ Iteration failed:", e)