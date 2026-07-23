import re
import sys
from collections import defaultdict

def format_stage_name(original_name):
    """
    Parses specific stage names (DiT forward/backward, VAE forward) into a 
    more readable format. If the name doesn't match a known pattern, 
    it returns the original name.
    """
    # Regex for DiT forward: forward_single<..._bs_..._f_..._h_..._w_..._sp<...>
    dit_forward_pattern = re.match(r'forward_single\d+_bs_(\d+)_f_(\d+)_h_(\d+)_w_(\d+)_sp(\d+)', original_name)
    if dit_forward_pattern:
        bs, f, h, w, sp = dit_forward_pattern.groups()
        return f"DiT forward (bs={bs}, f={f}, h={h}, w={w}), SP={sp}"

    # Regex for DiT backward: backward_bs_..._f_..._h_..._w_..._sp_<...>
    dit_backward_pattern = re.match(r'backward_bs_(\d+)_f_(\d+)_h_(\d+)_w_(\d+)_sp_(\d+)', original_name)
    if dit_backward_pattern:
        bs, f, h, w, sp = dit_backward_pattern.groups()
        return f"DiT backward (bs={bs}, f={f}, h={h}, w={w}), SP={sp}"
    
    # --- NEW: Regex for VAE forward ---
    # Matches vae_forward_bs_..._f_..._h_..._w_..._sp_..._dist<...>
    vae_forward_pattern = re.match(r'vae_forward_bs_(\d+)_f_(\d+)_h_(\d+)_w_(\d+)_sp_(\d+)_dist(.*)', original_name)
    if vae_forward_pattern:
        bs, f, h, w, sp, dist_tag = vae_forward_pattern.groups()
        # Clean up dist_tag if it's empty
        dist_tag = dist_tag if dist_tag else "N/A"
        return f"VAE forward (bs={bs}, f={f}, h={h}, w={w}), SP={sp}, dist={dist_tag}"

    # If no specific pattern matches, return the original name
    return original_name

def generate_timeline_from_log(log_file_path):
    """
    Parses a log file, groups entries by iteration and rank,
    and prints a timeline of stages for each group in SECONDS.
    """
    # Regex to capture Iteration, Rank, Stage Name, and CUDA time
    log_pattern = re.compile(
        r'\[Iter\s*(\d+)\]\[Rank\s*(\d+)\]\s*Stage:\s*(.*?)\s*\|.*CUDA:\s*([\d.]+)'
    )

    grouped_stages = defaultdict(list)

    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = log_pattern.match(line)
                if match:
                    iteration = int(match.group(1))
                    rank = int(match.group(2))
                    stage_name = match.group(3).strip()
                    # Convert ms to seconds immediately after parsing
                    cuda_time_s = float(match.group(4)) / 1000.0
                    
                    if stage_name.startswith('iteration-'):
                        continue

                    grouped_stages[(iteration, rank)].append({
                        'name': stage_name,
                        'cuda_time_s': cuda_time_s
                    })
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    if not grouped_stages:
        print("No valid log entries were found in the specified format.")
        return

    sorted_keys = sorted(grouped_stages.keys())

    for iteration, rank in sorted_keys:
        print("\n" + "="*80)
        print(f" Timeline for Iteration: {iteration}, Rank: {rank}")
        print("="*80)
        print(f"{'Start (s)':>15s} -> {'End (s)':>15s} | {'Duration (s)':>15s} | Stage Name")
        print(f"{'-'*15} -> {'-'*15} | {'-'*15} | {'-'*50}")

        current_time_s = 0.0
        stages = grouped_stages[(iteration, rank)]

        for stage in stages:
            duration_s = stage['cuda_time_s']
            start_time_s = current_time_s
            end_time_s = start_time_s + duration_s
            
            # Format the stage name before printing
            formatted_name = format_stage_name(stage['name'])
            
            print(f"{start_time_s:>15.3f} -> {end_time_s:>15.3f} | {duration_s:>15.3f} | {formatted_name}")
            
            current_time_s = end_time_s
            
    print("\n" + "="*80)
    print("Log processing complete.")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_logs.py <path_to_your_log_file>")
        print("\nDemonstrating with an example log file named 'sample.log'...")
        # Updated dummy content to include the new VAE format
        dummy_log_content = """[Iter 0][Rank 10] Stage: vae_forward_bs_1_f_69_h_1280_w_720_sp_8_distbaseline | CPU: 28475.448ms, CUDA: 28474.951ms
[Iter 0][Rank 10] Stage: forward_single20_bs_1_f_69_h_1280_w_720_sp8 | CPU: 14358.876ms, CUDA: 14358.647ms
[Iter 0][Rank 10] Stage: backward_bs_1_f_69_h_1280_w_720_sp_8 | CPU: 7665.294ms, CUDA: 7665.201ms
[Iter 0][Rank 10] Stage: iteration-0                       | CPU: 98585.304ms, CUDA: 98583.344ms
[Iter 1][Rank 10] Stage: vae_forward_bs_1_f_109_h_1280_w_720_sp_6_distdynamic | CPU: 2763.581ms, CUDA: 2763.521ms
[Iter 1][Rank 10] Stage: get_and_move_data_overhead       | CPU:  553.574ms, CUDA:  553.585ms
[Iter 1][Rank 10] Stage: forward_single20_bs_1_f_109_h_1280_w_720_sp6 | CPU: 13256.814ms, CUDA: 13257.274ms
[Iter 1][Rank 10] Stage: iteration-1                       | CPU: 46780.165ms, CUDA: 46779.109ms
"""
        with open("sample.log", "w") as f:
            f.write(dummy_log_content)
        
        generate_timeline_from_log("sample.log")
    else:
        log_file = sys.argv[1]
        generate_timeline_from_log(log_file)