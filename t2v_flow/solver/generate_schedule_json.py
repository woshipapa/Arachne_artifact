import json

def generate_schedule_json():
    """
    根据处理时间和调度配置，生成任务调度的JSON文件。
    """
    # 1. 从您提供的图片中提取的原始处理时间数据
    # 我已将其转换为标准的Python字典格式，其中SP作为键
    proc_times = {
        'D_97': {'VAE': {1: 41.2, 2: 25.1, 4: 14.4, 8: 8.8}, 'DiT': {1: 29.84, 2: 15.26, 4: 7.73, 8: 3.94}},
        'D_121': {'VAE': {1: 47.6, 2: 28.04, 4: 15.7, 8: 10.03}, 'DiT': {1: 45.2, 2: 22.81, 4: 11.55, 8: 5.83}},
        'D_221': {'VAE': {2: 52.9, 4: 26.64, 8: 16.5}, 'DiT': {2: 74.8, 4: 36.5, 8: 18.36}},
        'D_241': {'VAE': {2: 52.26, 4: 30.01, 8: 17.29}, 'DiT': {2: 88.61, 4: 42.7, 8: 21.58}}
    }

    # 2. 根据您的要求定义的调度分配规则
    schedule_config = [
        {'model': 'D_241', 'sp': 4, 'ranks': [0, 1,2,3]},
        {'model': 'D_221', 'sp': 2, 'ranks': [4,5]},
        {'model': 'D_97',  'sp': 1, 'ranks': [6]},
        {'model': 'D_121', 'sp': 1, 'ranks': [7]},
    ]
    
    # 3. 计算时间轴并生成任务列表
    schedule_list = []
    # 使用一个字典来追踪每组rank的下一个可用时间
    # key是ranks的元组(因为列表不能做字典的key)，value是时间
    next_available_time_on_ranks = {}

    print("开始生成调度计划...")
    
    for config in schedule_config:
        model_id = config['model']
        sp = config['sp']
        ranks = config['ranks']
        ranks_key = tuple(ranks) # 将列表转为元组用作key

        # 初始化这组ranks的开始时间
        if ranks_key not in next_available_time_on_ranks:
            next_available_time_on_ranks[ranks_key] = 0.0

        current_time = next_available_time_on_ranks[ranks_key]

        # 依次处理VAE和DiT两个模块
        for module in ['VAE', 'DiT']:
            task_name = f"{model_id}_{module}"
            
            # 查找处理时间
            try:
                processing_time = proc_times[model_id][module][sp]
            except KeyError:
                print(f"错误：在proc_times中找不到 {model_id} -> {module} -> SP={sp} 的时间，跳过此任务。")
                continue

            # 计算开始和结束时间
            start_time = current_time
            end_time = start_time + processing_time
            
            # 创建任务字典
            task = {
                "name": task_name,
                "sp": sp,
                "start": round(start_time, 2), # 保留两位小数
                "end": round(end_time, 2),
                "ranks": ranks
            }
            schedule_list.append(task)
            
            print(f"  - 已安排任务: {task_name}, Ranks: {ranks}, 时间: {start_time:.2f} -> {end_time:.2f}")

            # 更新当前时间，为下一个任务做准备
            current_time = end_time
        
        # 更新这组ranks的下一个可用时间
        next_available_time_on_ranks[ranks_key] = current_time

    # 4. 将生成的任务列表写入JSON文件
    output_filename = "schedule_data.json"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(schedule_list, f, ensure_ascii=False, indent=4)
        print(f"\n成功！调度数据已生成并保存到文件: {output_filename}")
    except Exception as e:
        print(f"\n错误：保存文件时出错: {e}")

    return schedule_list

# --- 运行主程序 ---
if __name__ == "__main__":
    generate_schedule_json()