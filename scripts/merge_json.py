import json

def merge_and_average():
    file1 = "iteration_times_summary.json"
    file2 = "makespan_stats.json"

    with open(file1, "r") as f1, open(file2, "r") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    merged = {}
    # 遍历两个文件的公共 key
    for key in data1.keys():
        if key in data2:
            merged[key] = (float(data1[key]) + float(data2[key])) / 2.0

    # 保存结果
    with open("merged_avg.json", "w") as f:
        json.dump(merged, f, indent=4)

    print("合并完成，结果已保存到 merged_avg.json")
    print(json.dumps(merged, indent=4))

if __name__ == "__main__":
    merge_and_average()
