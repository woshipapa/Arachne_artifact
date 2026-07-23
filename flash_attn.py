import torch
import torch.nn.functional as F

#(1,121861,6,128)
#(1,6,121861,128)
query = torch.rand(1,6,121861,128).to("cuda")
key = torch.rand(1,6,121861,128).to("cuda")
value = torch.rand(1,6,121861,128).to("cuda")

def trace_handler(p):
    p.export_chrome_trace(f"/root/deepspeed_baseline/flash_profile/step_{p.step_num}.json")
        
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=3),
    with_stack=True,
    on_trace_ready=trace_handler,
    record_shapes=True
) as prof:
    for i in range(10):
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        prof.step()