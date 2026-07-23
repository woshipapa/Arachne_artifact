import re
from typing import Dict, List, TextIO
import threading

# SingletonMeta 类保持不变
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

    # <-- 关键改动 1: 在构造函数中接收要处理的 rank 列表 -->
    def __init__(self, file_path: str, ranks_to_process: List[int] = None, initial_batch_size: int = 100):
        self.file_path = file_path
        self._file_handle: TextIO = None
        self.data: Dict[int, Dict[str, Dict]] = {}
        self.batch_size = initial_batch_size
        
        # 如果用户没有指定，则默认处理 ranks 0, 2, 4, 6
        if ranks_to_process is None:
            self.ranks_to_process = [0, 2, 4, 6]
        else:
            self.ranks_to_process = ranks_to_process
        print(f"Parser将处理以下Ranks: {self.ranks_to_process}")

        try:
            self._file_handle = open(self.file_path, 'r', encoding='utf-8')
        except FileNotFoundError:
            print(f"错误: 文件 '{self.file_path}' 未找到。")
            return
            
    def _parse_line(self, line: str) -> Dict[str, List[int]]:
        # 这个函数保持不变，它解析 shape 的逻辑是正确的
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

    # <-- 关键改动 2: _format_iteration_data 现在使用 self.ranks_to_process -->
    # all_ranks_data 现在是一个字典 {rank_num: data}
    def _format_iteration_data(self, all_ranks_data: Dict[int, Dict]) -> Dict[str, Dict]:
        processed_iteration = {}
        # 遍历我们在初始化时指定的 ranks 列表
        for rank_index in self.ranks_to_process:
            # 从解析好的字典中获取对应 rank 的数据
            rank_data = all_ranks_data.get(rank_index)
            
            if rank_data: # 如果这个rank的数据存在
                image_shape = rank_data.get('images')
                if not image_shape or not isinstance(image_shape, list) or len(image_shape) < 2:
                    continue

                # 你的形状修改逻辑
                if image_shape[1] > 129:
                    image_shape[1] = 129

                # 生成唯一的动态键名
                if len(image_shape) >= 5:
                    base_key_name = f"{image_shape[0]}_{image_shape[1]}_{image_shape[3]}_{image_shape[4]}"
                    final_key_name = f"{base_key_name}_rank{rank_index}"
                    processed_iteration[final_key_name] = rank_data
        
        return processed_iteration

    # <-- 关键改动 3: read_next_batch 现在更健壮，直接解析 Rank 编号 -->
    def read_next_batch(self, batch_size: int = 100) -> Dict[int, Dict]:
        if not self._file_handle or self._file_handle.closed:
            return {}

        newly_read_data = {}
        iterations_counted = 0
        current_iteration_num = None
        # 将 current_iteration_ranks 从列表改为字典，存储 {rank_num: shapes}
        current_iteration_ranks = {}

        # 正则表达式来匹配包含 Rank 编号的行
        line_pattern = re.compile(r"\[Iter (\d+) \| Rank (\d+)\]")

        for line in self._file_handle:
            header_match = re.search(r"==================== Iteration (\d+) ====================", line)
            
            if header_match:
                # 当遇到新的 Iteration 标志时，处理上一个完整 Iteration 的数据
                if current_iteration_num is not None and current_iteration_ranks:
                    processed_data = self._format_iteration_data(current_iteration_ranks)
                    self.data[current_iteration_num] = processed_data
                    newly_read_data[current_iteration_num] = processed_data
                    iterations_counted += 1
                    
                    if iterations_counted >= batch_size:
                        # 在返回前，设置好下一个 iteration 的状态
                        current_iteration_num = int(header_match.group(1))
                        current_iteration_ranks = {}
                        return newly_read_data

                # 开始一个新的 Iteration
                current_iteration_num = int(header_match.group(1))
                current_iteration_ranks = {}
                continue

            # 如果是数据行，解析它
            data_line_match = line_pattern.search(line)
            if data_line_match:
                iter_num_in_line = int(data_line_match.group(1))
                rank_num = int(data_line_match.group(2))

                # 确保这行数据属于当前正在处理的 Iteration
                if iter_num_in_line == current_iteration_num:
                    parsed_shapes = self._parse_line(line)
                    if parsed_shapes:
                        current_iteration_ranks[rank_num] = parsed_shapes

        # 文件读取结束后，处理最后一个 Iteration
        if current_iteration_num is not None and current_iteration_num not in self.data and current_iteration_ranks:
            processed_data = self._format_iteration_data(current_iteration_ranks)
            self.data[current_iteration_num] = processed_data
            newly_read_data[current_iteration_num] = processed_data

        return newly_read_data

    # get_iteration, get_all_loaded_data, close, __del__ 方法保持不变
    def get_iteration(self, iteration_number: int) -> Dict[str, Dict]:
        return self.data.get(iteration_number)

    def get_all_loaded_data(self) -> Dict[int, Dict]:
        return self.data

    def close(self):
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.close()

    def __del__(self):
        self.close()