import json
from collections import OrderedDict

def generate_bucket_config(oom_file='oom.json', output_file='bucket_config.py'):
    """

    Args:
    """
    try:
        with open(oom_file, 'r') as f:
            oom_data_str_keys = json.load(f)
        oom_data = {int(k): v for k, v in oom_data_str_keys.items()}
        print(f"Read and parsed {oom_file}")
    except FileNotFoundError:
        print(f"Error: {oom_file} not found. Make sure it exists in the current directory.")
        return
    except json.JSONDecodeError:
        print(f"Error: {oom_file} is malformed and cannot be parsed.")
        return

    frame_buckets = list(range(5, 130, 4))
    print(f"Frame buckets generated: {frame_buckets[0]} to {frame_buckets[-1]}, step 4")

    new_bucket_config = OrderedDict()

    for frame_count in frame_buckets:
        max_bs = 0
        for bs, max_frames in oom_data.items():
            if max_frames >= frame_count:
                if bs > max_bs:
                    max_bs = bs
        
        if max_bs > 0:
            new_bucket_config[frame_count] = (1.0, max_bs)
        else:
            print(f"Warning: no supported batch size found in {oom_file} for {frame_count} frames.")

    output_content = 'bucket_config = {\n'
    output_content += '    "960px": {\n'
    for frame, (val1, val2) in new_bucket_config.items():
        output_content += f'        {frame}: ({val1}, {val2}),\n'
    output_content += '    },\n'
    output_content += '}\n'

    with open(output_file, 'w') as f:
        f.write(output_content)
    
    print(f"Configuration written to: {output_file}")
    print("\n--- preview of the generated file ---")
    print(output_content)


if __name__ == '__main__':
    model = "hunyuan"
    resolution = "720p"
    generate_bucket_config(oom_file=f"t2v_flow/predictor/oom_thresholds/hunyuan/oom_thresholds_{model}_{resolution}_sp4.json",
                           output_file=f"bucket_config_for_{model}_{resolution}_sp4.py")