import os 
import argparse 
import torch
import torch.nn.functional as F 
from diffusers.models.attention_processor import Attention




def run_forward(args, attn_module):
    attn = attn_module
    hidden_states = args['hidden_states']
    encoder_hidden_states = args['encoder_hidden_states']
    image_rotary_emb = args['image_rotary_emb']
    attention_mask = args['attention_mask']

    print(hidden_states)
    print(attention_mask)
    if attn.add_q_proj is None and encoder_hidden_states is not None:
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    # 1. QKV projections
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    # print(f'query shape is {query.shape}')
    query = query.unflatten(2, (24, -1)).transpose(1, 2)
    key = key.unflatten(2, (24, -1)).transpose(1, 2)
    value = value.unflatten(2, (24, -1)).transpose(1, 2)

    # 2. QK normalization
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # 3. Rotational positional embeddings applied to latent stream
    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        if attn.add_q_proj is None and encoder_hidden_states is not None:
            query = torch.cat(
                [
                    apply_rotary_emb(
                        query[:, :, : -encoder_hidden_states.shape[1]],
                        image_rotary_emb,
                    ),
                    query[:, :, -encoder_hidden_states.shape[1] :],
                ],
                dim=2,
            )
            key = torch.cat(
                [
                    apply_rotary_emb(
                        key[:, :, : -encoder_hidden_states.shape[1]],
                        image_rotary_emb,
                    ),
                    key[:, :, -encoder_hidden_states.shape[1] :],
                ],
                dim=2,
            )
        else:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

    # 5. Attention
    torch.backends.cuda.enable_cudnn_sdp(True)
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)

    # 6. Output projection
    if encoder_hidden_states is not None:
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : -encoder_hidden_states.shape[1]],
            hidden_states[:, -encoder_hidden_states.shape[1] :],
        )

    return hidden_states, encoder_hidden_states


def run_backward(output, args):
    hidden_states = output[0]
    hidden_states.backward(args['grad_output'])
    grads = {
        "hidden_states": args['hidden_states'].grad
    }
    return grads 


def build_module(args):
    attn = Attention(
            query_dim=3072,
            cross_attention_dim=None,
            dim_head=128,
            heads=24,
            out_dim=3072,
            bias=True,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )
    attn.load_state_dict(torch.load(args.weight)).bfloat16()
    if args.device == "gpu":
        attn.cuda()
    return attn


def load_tensors(args):
    inputs = torch.load(args.input)[2]
    inputs['grad_output'] = torch.load(args.grad_output)[1][0]
    if args.device == "cpu":
        for k in inputs:
            inputs[k] = inputs[k].cpu()
    return inputs 


def save_tensors(tensors, save_dir):
    for k in tensors:
        save_path = os.path.join(save_dir, f"{k}.pt")
        torch.save(tensors[k], save_path)


def set_requires_grad(tensors):
    tensors['hidden_states'].requires_grad_()
    tensors['hidden_states'].retain_grad()


def test_attn_api(tensors, save_dir):
    set_requires_grad(tensors)
    output = run_forward(tensors)
    grads = run_backward(output, tensors)
    save_tensors(grads, save_dir)
    return grads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-w", "--weight")
    parser.add_argument("-g", "--grad_output")
    parser.add_argument("-d", "--device", choices=["cpu", "gpu"])
    parser.add_argument("-s", "--save_dir")
    args = parser.parse_args()
    tensors = load_tensors(args)
    attn_module = build_module(args)
    output = run_forward(tensors, attn_module)
    grads = run_backward(output, tensors)
    # print(grads['hidden_states'].max(), grads['hidden_states'].min(), grads['hidden_states'].norm(), flush=True)
    # print("aaaaaaaaaaaaaaaaaaaaaaaa")
    save_tensors(grads, args.save_dir)



if __name__ == "__main__":
    main()
