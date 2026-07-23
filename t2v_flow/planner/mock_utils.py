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
        task_list = []

        main_gpus = gpus
        sp = len(main_gpus)

        all_gpus = list(range(total_gpus))
        used_gpus = set(main_gpus)
        remaining_gpus = [i for i in all_gpus if i not in used_gpus]

        remaining_gpus += [i for i in gpus[sp:] if i not in used_gpus]
        remaining_gpus = sorted(set(remaining_gpus))

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

        content = yaml.dump({"tasks": task_list}, sort_keys=False)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[OK] YAML written to: {output_path}")
        return output_path
