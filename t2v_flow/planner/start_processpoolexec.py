from .schedule_pool_no_threading import SchedulePool
import cost_model_config


import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict


TOTAL_CPU = os.cpu_count() or 64
OUTER_WORKERS = 1
NGPUS = 16
SCHEDULE_TYPE = "genetic"
MACHINE_SIZE = 8

CORES_PER_ITER = max(1, (TOTAL_CPU - 2) // OUTER_WORKERS)


# =========================
# =========================
def run_one_iteration(
    iteration: int,
    pool_kwargs: Dict,
    cores_per_iter: int,
):

    joblib_tmp = f"/tmp/joblib_iter_{iteration}_pid_{os.getpid()}"
    os.makedirs(joblib_tmp, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = joblib_tmp

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    os.environ["GA_NUM_CORES"] = str(cores_per_iter)

    pool = SchedulePool(**pool_kwargs)
    pool.submit(iteration)

    path = os.path.join(pool.output_dir, f"schedule_{iteration}.yaml")
    return path


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    iterations = list(range(50))

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