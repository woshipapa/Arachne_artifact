# config.py

BASE_PATH = "t2v_flow/predictor"


VAE_MODEL_FILE = f"{BASE_PATH}/tile_poly/merged_model_parameters_FINAL.json"

DIT_FORWARD_MODEL_FILE = f"{BASE_PATH}/dit_poly/dit_predictor_1248_gradient_boosting.joblib"
DIT_BACKWARD_MODEL_DIR = f"{BASE_PATH}/dit_poly/sp_specific_models/"

INITIAL_BATCH_SIZE = 100
# INITIAL_BATCH_SIZE = 1
MODEL_STR = "hunyuan"
RESOLUTION = "720p"
MAX_FRAMES = 129
CLUSTER_STR = ""
# if RESOLUTION == "1080p":
#     MAX_FRAMES = 57
def get_simulation_log_path():
    if INITIAL_BATCH_SIZE == 1:
        return f"simulation_data/simulation_log_{MODEL_STR}_case.txt"
    if CLUSTER_STR != "":
        return f"simulation_data/simulation_log_{MODEL_STR}_{RESOLUTION}_{CLUSTER_STR}_{MAX_FRAMES}.txt"
    return f"simulation_data/simulation_log_{MODEL_STR}_{RESOLUTION}_{MAX_FRAMES}.txt"
MODELS_BASE_PATH = f"trained_models"
BASE_MODEL_NAME = "tile_encoder_base_model_sp1"
SYSTEM_MODEL_NAME = "system_forward_model_multi_sp"

DIT_MODEL_SP_MAP = {
        "wan": [1, 2, 3, 4, 6, 12],
        "hunyuan": [2, 4, 6, 8, 12], ## 32/64 gpus
        # "hunyuan": [2, 4, 6, 8, 12],
        "cogvideox": [1,2,3,5,6,10]
}


OOM_THRESHOLDS_DIR = f"{BASE_PATH}/oom_thresholds"