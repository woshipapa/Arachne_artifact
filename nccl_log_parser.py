import re
import argparse
from collections import defaultdict

def parse_log_file(log_file_path):
    comm_init_pattern = re.compile(
        r"\[(\d+)\]\s+NCCL INFO comm (\S+)\s+rank\s+\d+\s+nRanks\s+(\d+)\s+nNodes\s+(\d+)"
    )
    topology_pattern = re.compile(
        r"\[(\d+)\]\s+NCCL INFO (Ring|Tree)\s+(\S+)\s*:\s*(.*)$"
    )
    nccl_op_pattern = re.compile(
        r"\[(\d+)\]\s+NCCL INFO (Send|Recv):\s+.*?count\s+(\d+)\s+.*?root\s+(\d+)"
    )
    layer_timing_pattern = re.compile(
        r"\[Rank\s+(\d+)\].*?\[Iter\s+(\d+)\].*?Stage\s+'(.*?)':\s*CPU\s+(.*?),\s*CUDA\s+(.*?)$"
    )
    nccl_channel_pattern = re.compile(
        r"\[(\d+)\]\s+NCCL INFO Channel\s+"
        r"(\S+)\s*:\s*"
        r"(\S+\s*->\s*\S+)\s+"
        r"(?:\[(send|recv)\]\s*)?"
        r"via\s+(.*)$"
    )
    parallel_state_pattern = re.compile(
        r"\[Rank\s+(\d+)\]"
        r".*?ParallelState:\s*(.*)$"
    )
    
    parsed_data = defaultdict(list)
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                nccl_op_match = nccl_op_pattern.search(line)
                if nccl_op_match:
                    rank_id = int(nccl_op_match.group(1)); event_data = {'op_type': nccl_op_match.group(2).strip(), 'count': int(nccl_op_match.group(3)), 'root': int(nccl_op_match.group(4))}; parsed_data[rank_id].append({'line_num': i, 'type': 'NCCL_OP', 'data': event_data}); continue
                layer_match = layer_timing_pattern.search(line)
                if layer_match:
                    rank_id = int(layer_match.group(1)); event_data = {'iter': layer_match.group(2).strip(), 'stage_name': layer_match.group(3).strip(), 'cpu_time': layer_match.group(4).strip(), 'cuda_time': layer_match.group(5).strip()}; parsed_data[rank_id].append({'line_num': i, 'type': 'LAYER_TIMING', 'data': event_data}); continue
                topo_match = topology_pattern.search(line)
                if topo_match:
                    rank_id = int(topo_match.group(1)); event_data = {'topo_type': topo_match.group(2), 'topo_id': topo_match.group(3), 'details': topo_match.group(4).strip()}; parsed_data[rank_id].append({'line_num': i, 'type': 'TOPOLOGY', 'data': event_data}); continue
                comm_match = comm_init_pattern.search(line)
                if comm_match:
                    rank_id = int(comm_match.group(1)); event_data = {'comm_id': comm_match.group(2), 'total_ranks': comm_match.group(3), 'total_nodes': comm_match.group(4)}; parsed_data[rank_id].append({'line_num': i, 'type': 'COMM_INIT', 'data': event_data}); continue
                nccl_match = nccl_channel_pattern.search(line)
                if nccl_match:
                    rank_id = int(nccl_match.group(1)); event_data = {'channel_id': nccl_match.group(2).strip(), 'connection': nccl_match.group(3).strip(), 'type': (nccl_match.group(4) or 'N/A').strip(), 'via': nccl_match.group(5).strip()}; parsed_data[rank_id].append({'line_num': i, 'type': 'NCCL_CHANNEL', 'data': event_data}); continue
                ps_match = parallel_state_pattern.search(line)
                if ps_match:
                    rank_id = int(ps_match.group(1)); event_data = {'message': ps_match.group(2).strip()}; parsed_data[rank_id].append({'line_num': i, 'type': 'PARALLEL_STATE', 'data': event_data}); continue
    except FileNotFoundError:
        print(f"error: file not found '{log_file_path}'"); return None
    except Exception as e:
        print(f"error while parsing: {e}"); return None
    return parsed_data

def _print_aggregation_summary(buffer, first_line_num):
    total_ops = sum(d['count'] for d in buffer['send'].values()) + sum(d['count'] for d in buffer['recv'].values())
    if total_ops == 0: return
    print(f"L{first_line_num}:- [AGGR]    aggregated {total_ops} send/recv ops:")
    for op_type in ['send', 'recv']:
        if buffer[op_type]:
            sorted_sizes = sorted(buffer[op_type].keys())
            for size in sorted_sizes:
                data = buffer[op_type][size]; count = data['count']; peers = sorted(list(data['peers'])); size_mb = size / (1024*1024)
                print(f"       - {op_type.capitalize():<5}: Size {size} ({size_mb:.2f} MB) x {count}, Peers: {peers}")

def process_and_print_results(data, args):
    if not data:
        print("no valid data parsed."); return

    ranks_to_process = sorted(data.keys())
    if args.ranks is not None:
        specified_ranks = set(args.ranks)
        ranks_to_process = [r for r in ranks_to_process if r in specified_ranks]

    if not ranks_to_process:
        print(f"no events found for rank(s) {args.ranks}.")
        return
    # ==============================================================================

    for rank_id in ranks_to_process:
        print("="*120); print(f"--- Rank {rank_id} (Events Aggregated by Stage) ---"); print("="*120)
        events = data[rank_id]
        if not events:
            print("  (No relevant events found for this rank)\n"); continue
        
        agg_buffer = {'send': defaultdict(lambda: {'count': 0, 'peers': set()}), 'recv': defaultdict(lambda: {'count': 0, 'peers': set()})}
        first_op_line_num = -1

        for event in events:
            line_num_str = f"L{event['line_num']}:"
            if event['type'] == 'NCCL_OP':
                if first_op_line_num == -1: first_op_line_num = event['line_num']
                op = event['data']; op_type_key = op['op_type'].lower()
                agg_buffer[op_type_key][op['count']]['count'] += 1
                agg_buffer[op_type_key][op['count']]['peers'].add(op['root'])
            elif event['type'] == 'LAYER_TIMING':
                if args.show_aggregation: _print_aggregation_summary(agg_buffer, first_op_line_num)
                agg_buffer = {'send': defaultdict(lambda: {'count': 0, 'peers': set()}), 'recv': defaultdict(lambda: {'count': 0, 'peers': set()})}
                first_op_line_num = -1
                if args.show_stage:
                    timing = event['data']; print(f"{line_num_str:<6} [STAGE]   Iter {timing['iter']}, Stage '{timing['stage_name']}': CPU {timing['cpu_time']}, CUDA {timing['cuda_time']}")
            elif event['type'] == 'COMM_INIT' and args.show_topology:
                comm = event['data']; print(f"{line_num_str:<6} [COMM]    New Communicator {comm['comm_id']} (nRanks: {comm['total_ranks']}, nNodes: {comm['total_nodes']})")
            elif event['type'] == 'TOPOLOGY' and args.show_topology:
                topo = event['data']; print(f"{line_num_str:<6} [TOPO]    {topo['topo_type']} {topo['topo_id']}: {topo['details']}")
            elif event['type'] == 'NCCL_CHANNEL' and args.show_channel:
                ch = event['data']; print(f"{line_num_str:<6} [CHANNEL] Channel {ch['channel_id']:<6} | Connection: {ch['connection']:<18} | Via: {ch['via']}")
            elif event['type'] == 'PARALLEL_STATE' and args.show_pstate:
                msg = event['data']['message']; print(f"{line_num_str:<6} [PSTATE]  {msg}")

        if args.show_aggregation: _print_aggregation_summary(agg_buffer, first_op_line_num)
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NCCL log parser: per-stage aggregation, toggleable views, rank filtering.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("logfile", help="log file to parse")
    
    filter_group = parser.add_argument_group('filters')
    filter_group.add_argument('--ranks', type=int, nargs='+', help="display toggle")
    # ============================================================================
    
    display_group = parser.add_argument_group('display toggles (all on by default; --no-<option> disables)')
    display_group.add_argument('--no-aggregation', dest='show_aggregation', action='store_false', help="disable this view")
    display_group.add_argument('--no-topology', dest='show_topology', action='store_false', help="disable this view")
    display_group.add_argument('--no-stage', dest='show_stage', action='store_false', help="disable this view")
    display_group.add_argument('--no-channel', dest='show_channel', action='store_false', help="disable this view")
    display_group.add_argument('--no-pstate', dest='show_pstate', action='store_false', help="disable this view")
    
    args = parser.parse_args()
    
    parsed_log_data = parse_log_file(args.logfile)
    if parsed_log_data:
        process_and_print_results(parsed_log_data, args)