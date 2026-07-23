import os
import time
import traceback
from typing import List, Dict, Any
import json
import torch.distributed as dist


from t2v_flow.planner.scheduler import schedule
from simulation_data import IterationLogParser
from t2v_flow.predictor import DitPredictor, VaePredictor, DitMemoryPredictor, VaeSystemPredictor
import cost_model_config

import faulthandler, sys
faulthandler.enable(all_threads=True)

SCHEDULE_TYPE = "genetic"
USE_TOPOLOGY = True
ENABLE_OPTIMAL_SEARCH = True
ENABLE_BNB_ORACLE = False
OPTIMAL_MODE = "topology"  
NGPUS = 16
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

class SchedulePool:
    def __init__(
        self,
        num_workers: int = 1,
        schedule_type: str = "a_star",
        output_dir: str = "generated_schedules",
        n_gpus: int = 8,
        machine_size: int = None,
        use_topology: bool = True
    ):
        self.schedule_type = schedule_type
        self.use_topology = use_topology
        base_dir = os.path.dirname(os.path.abspath(__file__))


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
            sp = 2 # for 32 or 64 gpus
        )
        self.data_loader.read_next_batch()
        
        self.model_type = cost_model_config.MODEL_STR
        self.resolution = cost_model_config.RESOLUTION
        self.max_frames = str(cost_model_config.MAX_FRAMES)
        self.output_dir = os.path.join(base_dir, output_dir, self.model_type, self.resolution, 
                                       self.max_frames, str(self.n_gpus) ,schedule_type, "test")
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


    def submit(self, iteration: int):
        print(f"🔄 [MainProcess] Starting iteration {iteration}...")
        
        yaml_path = os.path.join(self.output_dir, f"schedule_{iteration}.yaml")
        
        try:
            if iteration % self.data_loader.batch_size == 0 and iteration > 0:
                self.data_loader.read_next_batch()

            iteration_data = self.data_loader.get_iteration(iteration)
            if not iteration_data:
                print(f"⚠️ Warning: No data found for iteration {iteration}")
                return

            key_list = iteration_data.keys()
            print(f"📋 Processing {len(key_list)} tasks for iteration {iteration}...")
            
            self.process_one(key_list, yaml_path)
            
        except Exception as e:
            print(f"❌ Schedule generation failed for schedule_{iteration}: {e}")
            error_details = traceback.format_exc()
            print(error_details)

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
            

        if ENABLE_BNB_ORACLE:
            from .bnb.bnb_oracle_sp import bnb_oracle_sp_v1_exact
            oracle = bnb_oracle_sp_v1_exact(
                tasks=tasks_for_scheduler,
                n_gpus=self.n_gpus,
                time_budget_sec=60, 
                log=True
            )

            oracle_json_path = yaml_path.replace(".yaml", "_bnb_oracle.json")
            with open(oracle_json_path, "w") as f:
                json.dump(oracle, f, indent=2)

            if oracle["best_makespan"] is not None:
                print(
                    f"[BnB Oracle] best={oracle['best_makespan']:.3f}s "
                    f"proven={oracle['proven_optimal']} "
                    f"nodes={oracle['stats']['nodes']}"
                )
            else:
                print("[BnB Oracle] No feasible solution found (unexpected)")

        if ENABLE_OPTIMAL_SEARCH:
            try:
                if OPTIMAL_MODE == "pure_sp":
                    from .optimal_solver_pure_sp_size import solve_optimal
                    optimal_result = solve_optimal(
                        tasks=tasks_for_scheduler,
                        n_gpus=self.n_gpus,
                        time_limit_sec= 1000,
                        log=True
                    )
                    topology_aware = False
                    solver_name = "pure_sp"
                elif OPTIMAL_MODE == "brute_force":
                    from .brute_force_solver import solve_brute_force
                    print(f">>> [Baseline] Starting Brute Force Search for anytime curve...", flush=True)
                    optimal_result = solve_brute_force(
                        tasks=tasks_for_scheduler,
                        n_gpus=self.n_gpus,
                        time_limit_sec=60, 
                        log=True,
                        time_scale=1000
                    )
                    topology_aware = False
                    solver_name = "Brute Force (DFS + Greedy)"
                elif OPTIMAL_MODE == "topology":
                    from .topology_baseline.optimal_solver_complete import solve_optimal_with_topology
                    from .TopologyModel import TopologyModel
                    ga_trace_csv_path = yaml_path.replace(".yaml", "_trace.csv")

                    if not os.path.exists(ga_trace_csv_path):
                        print(f"⚠️ GA trace CSV not found: {ga_trace_csv_path}")
                        ga_trace_csv_path = None
                    optimal_result = solve_optimal_with_topology(
                        tasks=tasks_for_scheduler,
                        n_gpus=self.n_gpus,
                        topology_model=TopologyModel(params_file='fit_results.json', machine_size=self.machine_size, enable_topology=True),
                        time_limit_sec=300,
                        log=True,
                    )
                    topology_aware = True
                    solver_name = "topology cp sovler"
                elif OPTIMAL_MODE == "topology_baseline":
                    from .topology_baseline.SA_solver import solve_topology_baseline
                    from .TopologyModel import TopologyModel
                    
                    topo = TopologyModel(params_file='fit_results.json', machine_size=self.machine_size, enable_topology=True)
                    
                    optimal_result = solve_topology_baseline(
                        tasks=tasks_for_scheduler,
                        n_gpus=self.n_gpus,
                        topology_model=topo,
                        time_limit_sec=60,
                        log=True
                    )
                elif OPTIMAL_MODE == "dfs_topology":
                    from .topology_baseline.dfs_topology_sovler import solve_dfs_topology
                    from .TopologyModel import TopologyModel
                    
                    topo = TopologyModel(params_file='fit_results.json', machine_size=self.machine_size, enable_topology=True)
                    
                    optimal_result = solve_dfs_topology(
                        tasks=tasks_for_scheduler,
                        n_gpus=self.n_gpus,
                        topology_model=topo,
                        time_limit_sec=60,
                        log=True
                    )
                    topology_aware = True
                    solver_name = "dfs topology"
                else:
                    raise ValueError(f"Unknown OPTIMAL_MODE={OPTIMAL_MODE}")
               
               
                print(">>> [Oracle] solve_optimal returned.", flush=True)
                if optimal_result["status"] == "INFEASIBLE":
                    print("⚠️ Optimal solver returned INFEASIBLE. Possibly over-constrained.")
            except Exception as e:
                print(f"❌ Optimal solver failed: {e}")
                import traceback
                traceback.print_exc()
                optimal_result = {
                    "status": "FAILED",
                    "optimal_makespan": None,
                }
                
            suffix = "_bruteforce" if OPTIMAL_MODE == "brute_force" else "_optimal"
            optimal_json_path = yaml_path.replace(".yaml", f"{suffix}.json")
            
            optimal_dump = {
                "meta": {
                    "n_gpus": self.n_gpus,
                    "num_tasks": len(tasks_for_scheduler),
                    "time_limit_sec": 1000 if OPTIMAL_MODE != "brute_force" else 60,
                    "solver": solver_name,
                    "topology_aware": topology_aware,
                    "mode": OPTIMAL_MODE
                },
                "result": optimal_result, 
            }

            with open(optimal_json_path, "w") as f:
                
                json.dump(optimal_dump, f, indent=2)

            print(f"📝 [Result] Solver result saved to {optimal_json_path}")
        else:
            print("Don't go cp optimal solver!")
        schedule(
            yaml_path=yaml_path,
            tasks=tasks_for_scheduler,
            n_gpus=self.n_gpus,
            type=self.schedule_type,
            machine_size=self.machine_size,
            use_topology=self.use_topology
        )

    def get(self, iteration: int, block: bool = True, timeout: float = None) -> str:
        path = os.path.join(self.output_dir, f"schedule_{iteration}.yaml")
        if os.path.exists(path):
            return path
        else:
            raise FileNotFoundError(f"Schedule file was not generated: {path}")

    def close(self):
        print("Scheduler pool closed.")


if __name__ == "__main__":

    pool = SchedulePool(
        num_workers=1,
        n_gpus=NGPUS,
        machine_size=8,
        use_topology=USE_TOPOLOGY,
        schedule_type=SCHEDULE_TYPE
    )
    
    for i in range(50):
        pool.submit(i)

    pool.close()
