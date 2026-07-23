import json

def merge_and_average():
    file1 = "iteration_times_summary.json"
    file2 = "makespan_stats.json"

    with open(file1, "r") as f1, open(file2, "r") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    merged = {}
    for key in data1.keys():
        if key in data2:
            merged[key] = (float(data1[key]) + float(data2[key])) / 2.0

    with open("merged_avg.json", "w") as f:
        json.dump(merged, f, indent=4)

    print("merged; saved to merged_avg.json")
    print(json.dumps(merged, indent=4))

if __name__ == "__main__":
    merge_and_average()
