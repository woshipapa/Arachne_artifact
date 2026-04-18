import yaml
import os


class MockUtils:
    @staticmethod
    def save_fixed_sp_yaml(
        name: str,
        gpus: list[int],
        output_path: str,
        total_gpus: int = 8,
        fill_task_prefix: str = "FILL_TASK",
        task_type: str = "FULL",
    ) -> str:
        """
        所有任务（包括主任务和填充任务）都使用固定 sp 数量的 GPU。

        :param name: 主任务名称
        :param gpus: 主任务请求的 GPU 列表（只取前 sp 个）
        :param output_path: 输出文件路径
        :param sp: 每个任务使用 GPU 数
        :param total_gpus: 总共的 GPU 数量
        :param fill_task_prefix: 填充任务前缀
        :param task_type: 主任务类型
        """
        task_list = []

        # 确定主任务 GPU 分配（只取前 sp 个）
        main_gpus = gpus
        sp = len(main_gpus)

        all_gpus = list(range(total_gpus))
        used_gpus = set(main_gpus)
        remaining_gpus = [i for i in all_gpus if i not in used_gpus]

        # 把主任务未用到的 GPU（gpus[sp:]）也加回剩余池中
        remaining_gpus += [i for i in gpus[sp:] if i not in used_gpus]
        remaining_gpus = sorted(set(remaining_gpus))  # 去重排序

        # 添加主任务
        task_list.append(
            {
                "name": name,
                "gpus": main_gpus,
                "dependencies": [],
                "args": {
                    "task_type": task_type,
                    "sp": sp,
                },
            }
        )

        # 填充任务，按 sp 一组分配
        task_idx = 0
        while len(remaining_gpus) >= sp:
            fill_gpus = remaining_gpus[:sp]
            remaining_gpus = remaining_gpus[sp:]

            task_list.append(
                {
                    "name": f"{fill_task_prefix}_{task_idx}",
                    "gpus": fill_gpus,
                    "dependencies": [],
                    "args": {
                        "task_type": task_type,
                        "sp": sp,
                    },
                }
            )
            task_idx += 1

        # 写入 YAML
        content = yaml.dump({"tasks": task_list}, sort_keys=False)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ YAML 已保存到: {output_path}")
        return output_path
