import re
from typing import Dict, List, TextIO
import threading

class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class IterationLogParser(metaclass=SingletonMeta):

    def __init__(self, file_path: str, initial_batch_size: int = 100, sp: int = 4):

        self.file_path = file_path
        self._file_handle: TextIO = None
        self.data: Dict[int, Dict[str, Dict]] = {}
        self.batch_size = initial_batch_size
        self.sp = sp
        try:
            self._file_handle = open(self.file_path, 'r', encoding='utf-8')
        except FileNotFoundError:
            print(f"error: file '{self.file_path}' not found.")
            return
            
        # self.read_next_batch(initial_batch_size)

    def _parse_line(self, line: str) -> Dict[str, List[int]]:
        shapes = {}
        pattern = re.compile(r"'(\w+)': torch\.Size\(\[([^\]]*)\]\)")
        shape_dict_str = line[line.find('{'):]
        matches = pattern.findall(shape_dict_str)
        
        for key, values_str in matches:
            if values_str:
                shapes[key] = [int(v.strip()) for v in values_str.split(',')]
            else:
                shapes[key] = []
        return shapes

    def _format_iteration_data(self, all_ranks_data: List[Dict]) -> Dict[str, Dict]:
        processed_iteration = {}
        
        # if mpu.get_context_parallel_world_size() == 4:
        # 16gpus
        
        if self.sp == 4:
            ranks_to_process = [0, 2, 4, 6]
        elif self.sp == 2:
        # cog
            ranks_to_process = [0, 1,2,3,4,5,6,7]
        elif self.sp == 8:
            ranks_to_process = [0, 2]
        elif self.sp == 1:
            ranks_to_process = range(0,16)
        for rank_index in ranks_to_process:
            if rank_index < len(all_ranks_data):
                rank_data = all_ranks_data[rank_index]
                image_shape = rank_data.get('images')

                if not image_shape or not isinstance(image_shape, list) or len(image_shape) < 2:
                    fallback_key = f"rank_{rank_index}_image_data_malformed"
                    processed_iteration[fallback_key] = rank_data
                    continue


                for key, shape_list in rank_data.items():
                    # if isinstance(shape_list, list) and len(shape_list) > 1:
                    #     if shape_list[1] > 129:
                    #         shape_list[1] = 129
                    if key == "prompt_embeds":
                        # to cogvideox
                        shape_list[1] = 226

                if len(image_shape) >= 5:
                    base_key_name = f"{image_shape[0]}_{image_shape[1]}_{image_shape[3]}_{image_shape[4]}"
                    final_key_name = f"{base_key_name}_rank{rank_index}"
                    
                    processed_iteration[final_key_name] = rank_data
                else:
                    fallback_key = f"rank_{rank_index}_image_shape_incomplete"
                    processed_iteration[fallback_key] = rank_data
                

        return processed_iteration

    def read_next_batch(self, batch_size: int = 100) -> Dict[int, Dict]:
        if not self._file_handle or self._file_handle.closed:
            print("file not open or already closed.")
            return {}

        newly_read_data = {}
        iterations_counted = 0
        current_iteration_num = None
        current_iteration_ranks = []

        for line in self._file_handle:
            header_match = re.search(r"Iteration (\d+)", line)
            
            if header_match:
                if current_iteration_num is not None:
                    processed_data = self._format_iteration_data(current_iteration_ranks)
                    self.data[current_iteration_num] = processed_data
                    newly_read_data[current_iteration_num] = processed_data
                    iterations_counted += 1
                    
                    if iterations_counted >= batch_size:
                        return newly_read_data

                current_iteration_num = int(header_match.group(1))
                current_iteration_ranks = []
                continue

            if current_iteration_num is not None and line.strip().startswith('[Iter'):
                parsed_shapes = self._parse_line(line)
                if parsed_shapes:
                    current_iteration_ranks.append(parsed_shapes)

        if current_iteration_num is not None and current_iteration_num not in self.data:
            processed_data = self._format_iteration_data(current_iteration_ranks)
            self.data[current_iteration_num] = processed_data
            newly_read_data[current_iteration_num] = processed_data
        print(f"self.data = {self.data}")
        return newly_read_data

    def get_iteration(self, iteration_number: int) -> Dict[str, Dict]:
        data = self.data.get(iteration_number)
        
        if data is None:
            print(f"note: iteration {iteration_number} not loaded yet; call read_next_batch().")
        else:
            print(f"\n--- fetching data for iteration {iteration_number} ---")
            if not data:
                print("  (no valid data for this iteration)")
            else:
                total_sum = 0
                for key, shape_dict in data.items():
                    print(f"  Key   : {key}")
                    print(f"  Shapes: {shape_dict}")
                    if 'images' in shape_dict:
                        images_shape = shape_dict['images']
                        if len(images_shape) >= 2:
                            total_sum += images_shape[0] * images_shape[1]
                print(f"  weighted frame sum over all 'images': {total_sum}")
                new_file_name = self.file_path.replace("simulation_log", "total_frames")
                with open(new_file_name, "a") as f:
                    f.write(f"iteration {iteration_number} : {total_sum}\n")
            print(f"--- done ---\n")
            
        return data

    def get_all_loaded_data(self) -> Dict[int, Dict]:
        return self.data

    def close(self):
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.close()
            print("file handle closed.")

    def __del__(self):
        self.close()