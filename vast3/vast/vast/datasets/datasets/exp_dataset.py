from .IterationLogParser import IterationLogParser
import torch
from torch.utils.data import Dataset
from typing import List


# # self.sample_map = [
#   # --- 来自 Iteration 0 的样本 ---
#   { "iteration": 0, "sample_key": "1_101..._rank0" }, # 索引 0
#   { "iteration": 0, "sample_key": "1_113..._rank4" }, # 索引 1
#   { "iteration": 0, "sample_key": "1_113..._rank6" }, # 索引 2
#   { "iteration": 0, "sample_key": "2_45..._rank2"  }, # 索引 3

#   # --- 来自 Iteration 1 的样本将从这里开始 ---
#   { "iteration": 1, "sample_key": "key_A_from_iter1" }, # 索引 4
#   { "iteration": 1, "sample_key": "key_B_from_iter1" }, # 索引 5
  
#   # ... 依此类推, 直到所有 Iterations 中所有符合条件的样本都被添加进来
# ] 
class ExpDataset:

    def __init__(self, log_file_path: str,  ranks_to_process: List[int] = None, initial_batch_size: int = 1000, **kwargs):
        """
        Dataset的构造函数。
        - log_file_path: 你要解析的日志文件路径。
        - initial_batch_size: 每次从文件中读取的迭代次数。
        """
        super().__init__()
        
        # 1. 初始化并获取 Parser 的单例实例
        # 它会自动处理文件的打开和数据的缓存
        print(f"[Rank {torch.distributed.get_rank()}] ExpDataset log file path = {log_file_path}")
        self.parser = IterationLogParser(file_path=log_file_path, ranks_to_process=ranks_to_process, initial_batch_size=initial_batch_size)

        # 2. 加载所有数据
        # 持续调用 read_next_batch 直到文件读完
        print("开始加载并解析所有日志数据...")
        while self.parser.read_next_batch(batch_size=initial_batch_size):
            # 这个循环会持续进行，直到read_next_batch返回一个空字典，表示文件已读完
            pass
        all_data = self.parser.get_all_loaded_data()
        print(f"所有数据加载完毕，共 {len(all_data)} 次迭代。")

        # 3. 创建一个扁平化的样本映射表 (这是核心)
        # self.sample_map 的每一项都唯一对应一个训练样本
        self.sample_map = []
        # 按迭代编号排序，保证每次运行顺序一致
        sorted_iteration_keys = sorted(all_data.keys())
        for iter_num in sorted_iteration_keys:
            iteration_samples = all_data[iter_num]
            # 同样对内部的key排序，保证确定性
            for sample_key in sorted(iteration_samples.keys()):
                # 映射表存储了足够的信息，以便将来可以反向查找
                self.sample_map.append({
                    "iteration": iter_num,
                    "sample_key": sample_key
                })
        
        print(f"数据扁平化处理完成，共找到 {len(self.sample_map)} 个独立样本。")

        # 初始化 transform
        self.transform = None

    def __len__(self) -> int:
        """
        返回数据集中样本的总数。
        """
        return len(self.sample_map)

    def __getitem__(self, idx: int) -> dict:
        """
        根据索引 idx 获取一个样本。
        这是 DataLoader 工作的地方。
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        # 1. 从我们的映射表中找到该索引对应的元信息
        sample_info = self.sample_map[idx]
        iter_num = sample_info["iteration"]
        sample_key = sample_info["sample_key"]

        # 2. 从 Parser 中获取原始样本数据（这是一个字典，比如 {'images': [1, 129, ...], ...}）
        # get_iteration 方法非常快，因为它只是从内存中的字典里查找
        iteration_data = self.parser.get_iteration(iter_num)
        sample_data = iteration_data[sample_key]

        # 3. (可选) 对样本应用数据增强或转换
        if self.transform:
            sample_data = self.transform(sample_data)

        # 4. 返回最终的样本
        # 你可以根据需要调整返回的字典结构
        return {
            "metadata": sample_info, # 可以把元数据也一并返回，方便调试
            "shapes": sample_data
        }

    # 实现 set_transform 以与 get_dataloader 兼容
    def set_transform(self, transform):
        self.transform = transform

    # 实现 filter (目前是空操作，但为了兼容性保留)
    def filter(self, **kwargs):
        # 你的数据已经是预处理好的形状，这个filter可能不需要做什么
        # 如果需要，可以在这里根据kwargs过滤self.sample_map
        print("MyCustomDataset.filter() called, but no filtering logic is implemented.")
        pass
