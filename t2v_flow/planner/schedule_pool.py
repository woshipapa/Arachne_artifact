import threading
import os
import time
import queue
import traceback
from typing import List, Dict, Any
import json
import torch.distributed as dist

from t2v_flow.planner.scheduler import schedule
from simulation_data import IterationLogParser
from t2v_flow.predictor import DitPredictor, VaePredictor, DitMemoryPredictor, VaeSystemPredictor
import cost_model_config

SCHEDULE_TYPE = "genetic"
NGPUS = 16


class SchedulePool:
    def __init__(
        self,
        num_workers: int = 4,
        schedule_type: str = "a_star",
        output_dir: str = "generated_schedules",
        n_gpus: int = 8,
        machine_size: int = None,
    ):
        self.num_workers = num_workers
        self.schedule_type = schedule_type
        base_dir = os.path.dirname(os.path.abspath(__file__))


        self.task_queue = queue.Queue()
        self._start_workers()

        self.n_gpus = n_gpus
        self.machine_size = machine_size
        if os.environ.get("MODEL_TYPE") is not None:
            cost_model_config.MODEL_STR = os.environ.get("MODEL_TYPE")
        if os.environ.get("RESOLUTION") is not None:
            cost_model_config.RESOLUTION = os.environ.get("RESOLUTION")
        if os.environ.get("MAX_FRAMES") is not None:
            cost_model_config.MAX_FRAMES = int(os.environ.get("MAX_FRAMES"))
        self.data_loader = IterationLogParser(
            file_path = cost_model_config.get_simulation_log_path(),
            initial_batch_size = cost_model_config.INITIAL_BATCH_SIZE,
            sp = 2 if cost_model_config.MODEL_STR == "cogvideox" else 4
        )
        self.data_loader.read_next_batch()
        
        self.model_type = cost_model_config.MODEL_STR
        self.resolution = cost_model_config.RESOLUTION
        self.max_frames = str(cost_model_config.MAX_FRAMES)
        self.output_dir = os.path.join(base_dir, output_dir, self.model_type, self.resolution, 
                                       self.max_frames, schedule_type, "test")
        os.makedirs(self.output_dir, exist_ok=True)


        model_specific_dir = os.path.join(
            cost_model_config.MODELS_BASE_PATH, 
            self.model_type,
            self.resolution
        )
        
        self.vae_predictor = VaeSystemPredictor(
                model_dir = model_specific_dir,
                base_model_name = cost_model_config.BASE_MODEL_NAME,
                system_model_name = cost_model_config.SYSTEM_MODEL_NAME
            )

            
        self.dit_predictor = DitPredictor(
            model_name = self.model_type,
            resolution = self.resolution
        )

        self.current_model_sp_list = cost_model_config.DIT_MODEL_SP_MAP.get(self.model_type, [])
        print("Initializing Predictors...")

        oom_thresholds_dir = os.path.join(
            cost_model_config.OOM_THRESHOLDS_DIR, self.model_type
        )
        self.dit_memory_predictor = DitMemoryPredictor(
            oom_files_dir = oom_thresholds_dir,
            model_sp_map=cost_model_config.DIT_MODEL_SP_MAP
        )
        print("All predictors initialized.")

    def _start_workers(self):
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker, name=f"SchedulerWorker-{i}", daemon=True)
            t.start()

    def _worker(self):
        while True:
            iteration = self.task_queue.get()
            print(f"===============================worker=================================")
            if iteration is None:
                # Mark the shutdown sentinel as done too, otherwise
                # close()'s task_queue.join() blocks forever waiting on it.
                self.task_queue.task_done()
                break
            
            yaml_path = os.path.join(self.output_dir, f"schedule_{iteration}.yaml")
            try:
                if iteration % self.data_loader.batch_size == 0 and iteration > 0:
                    self.data_loader.read_next_batch()

                key_list = self.data_loader.get_iteration(iteration).keys()
                print(f"[{threading.current_thread().name}] Processing iteration {iteration} with {len(key_list)} tasks.")
                
                self.process_one(key_list, yaml_path)
                
            except Exception as e:
                print(f"❌ Schedule generation failed for schedule_{iteration}: {e}")
                error_details = traceback.format_exc()
                print(error_details)
            finally:
                self.task_queue.task_done()

    def process_one(self, key_list: List[str], yaml_path: str):
        tasks_for_scheduler: List[Dict[str, Any]] = []

        for i, key in enumerate(key_list):
            try:
                base_key, rank_suffix = key.rsplit('_', 1)
                k = base_key.split("_")
                bs, frame, h, w = map(int, k)
                if h == 1936:
                    h = 1072
                if w == 1072:
                    w = 1936
            except (ValueError, IndexError) as e:
                print(f"⚠️ Warning: Could not parse key '{key}'. Skipping. Error: {e}")
                continue

            dit_sp_list = self.dit_memory_predictor.get_available_sp_list(
                model_name=cost_model_config.MODEL_STR,
                bs=bs,
                f=frame,
                h=h,
                w=w
            )
            
            vae_sp_set = set(self.current_model_sp_list)
            vae_sp_set.add(1)
            vae_sp_list = sorted(list(vae_sp_set))

            if not dit_sp_list:
                print(f"⚠️ Warning: No available SP found for DiT on task '{key}' due to memory constraints. Skipping.")
                continue

            task_data = {
                "task_id": i,
                "dataset_id": key,
                "dit": [],
                "vae": []
            }

            for sp in dit_sp_list:
                dit_time = self.dit_predictor.predict(bs=bs, frame=frame, h=h, w=w, sp=sp, layers=60)
                task_data["dit"].append((sp, dit_time))

            for sp in vae_sp_list:
                vae_time = self.vae_predictor.predict(bs=bs,f=frame, h=h, w=w, sp=sp)

                task_data["vae"].append((sp, vae_time))
            
            tasks_for_scheduler.append(task_data)


        if not tasks_for_scheduler:
            print("No tasks to schedule after filtering.")
            return
        try:
            tasks_json_path = yaml_path.replace(".yaml", "_tasks.json")
            with open(tasks_json_path, 'w', encoding='utf-8') as f:
                json.dump(tasks_for_scheduler, f, indent=4, ensure_ascii=False)
            print(f"✅ Tasks data for schedule saved to '{tasks_json_path}'")
        except Exception as e:
            print(f"❌ Failed to save tasks to JSON file: {e}")
        print(f"Generating schedule for {len(tasks_for_scheduler)} tasks at '{yaml_path}'")
        schedule(
            yaml_path=yaml_path,
            tasks=tasks_for_scheduler,
            n_gpus=self.n_gpus,
            type=self.schedule_type,
            machine_size=self.machine_size,
        )

    def submit(self, iteration: int):
        self.task_queue.put(iteration)

    def get(self, iteration: int, block: bool = True, timeout: float = None) -> str:
        start_time = time.time()
        path = os.path.join(self.output_dir, f"schedule_{iteration}.yaml")
        
        while not os.path.exists(path):
            if not block:
                raise FileNotFoundError(f"Schedule file not ready: {path}")
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for schedule file: {path}")
            time.sleep(0.1)
        return path

    def close(self):
        print("Closing all scheduler workers...")
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        self.task_queue.join()
        print("All workers have been closed.")

if __name__ == "__main__":

    pool = SchedulePool(
        num_workers=1,
        n_gpus=NGPUS,
        schedule_type=SCHEDULE_TYPE,
        machine_size=8,
    )
    pool.submit(13)


    pool.close()
