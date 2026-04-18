import os
import re
import csv
import json

def collect_makespan():
    results = {}
    # 匹配文件名 schedule_{iter}_trace.csv
    pattern = re.compile(r"schedule_(\d+)_trace\.csv")

    for filename in os.listdir("."):
        match = pattern.match(filename)
        if match:
            iter_num = match.group(1)
            with open(filename, "r") as f:
                reader = csv.DictReader(f)
                # 直接读取第一行的 Makespan
                first_row = next(reader)
                makespan = float(first_row["Makespan"])
                results[iter_num] = makespan

    # 保存结果为 JSON
    with open("makespan_stats.json", "w") as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    collect_makespan()
