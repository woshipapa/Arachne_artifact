import os 
import argparse 
import torch

def load_tensors(args):
    data = {}
    data['query'] = torch.load(args.query).requires_grad()
    data['key'] = torch.load(args.key).requires_grad()
    data['value'] = torch.load(args.value).requires_grad()
    data['attention_mask'] = torch.load(args.attention_mask)
    data['grad_output'] =  torch.load(args.grad_output)
    if args.device == "cpu":
        for k in data:
            data[k] = data[k].cpu()
    # for k in data:
    #     data[k].requires_grad()
    return data 


def run_forward(tensors):
    output = torch.nn.functional.scaled_dot_product_attention(
        tensors['query'], tensors['key'], tensors['value'], attn_mask=tensors['attention_mask']
    )
    return output 


def run_backward(output, tensors):
    output.backward(tensors['grad_output'])
    grads = {
        "query": tensors['query'].grad,
        "key": tensors['key'].grad,
        "value": tensors['value'].grad
    }
    return grads 


def save_tensors(tensors, save_dir):
    for k in tensors:
        save_path = os.path.join(save_dir, f"{k}.pt")
        torch.save(tensors[k], save_path)


def set_requires_grad(tensors):
    tensors['query'].requires_grad_()
    tensors['key'].requires_grad_()
    tensors['value'].requires_grad_()
    tensors['query'].retain_grad()
    tensors['key'].retain_grad()
    tensors['value'].retain_grad()
    # tensors['query'] = tensors['query'] * 1
    # tensors['key'] = tensors['key'] * 1
    # tensors['value'] = tensors['value'] * 1


def test_sdpa_api(tensors, save_dir):
    set_requires_grad(tensors)
    output = run_forward(tensors)
    grads = run_backward(output, tensors)
    save_tensors(grads, save_dir)
    return grads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query")
    parser.add_argument("-k", "--key")
    parser.add_argument("-v", "--value")
    parser.add_argument("-m", "--attention_mask")
    parser.add_argument("-g", "--grad_output")
    parser.add_argument("-d", "--device", choices=["cpu", "gpu"])
    parser.add_argument("-s", "--save_dir")
    args = parser.parse_args()
    tensors = load_tensors(args)
    output = run_forward(tensors)
    grads = run_backward(output, tensors)
    save_tensors(grads, args.save_dir)



if __name__ == "__main__":
    main()
