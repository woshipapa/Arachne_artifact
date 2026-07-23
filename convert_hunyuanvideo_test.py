from diffusers import HunyuanVideoTransformer3DModel
import torch

# model = HunyuanVideoTransformer3DModel.from_pretrained("<CKPT_ROOT>/ckpt_2040_hf")

# model_origin = HunyuanVideoTransformer3DModel.from_pretrained("<CKPT_ROOT>/transformer_safetensor")

model = HunyuanVideoTransformer3DModel.from_pretrained("<CKPT_ROOT>/ckpt_2040_hf_iter_0000800_test", use_safetensors=False)

# print(model)

# model = torch.load("<CKPT_ROOT>/diffusion_pytorch_model.bin")
# model = torch.load("<CKPT_ROOT>/diffusion_pytorch_model.bin")
# for name, param in model.items():
#     print(f"{name}: {param.shape}")
# exit(0)

sd1 = model.state_dict()
sd2 = model_origin.state_dict()

keys1 = set(sd1.keys())
keys2 = set(sd2.keys())

different_keys = []

if keys1 != keys2:
    print("Key sets are different!")
    print("In sd1 not in sd2:", keys1 - keys2)
    print("In sd2 not in sd1:", keys2 - keys1)
else:
    print("All keys matched. Now comparing values...\n")

    all_same = True
    for key in keys1:
        tensor1 = sd1[key]
        tensor2 = sd2[key]
        if not torch.equal(tensor1, tensor2):
            different_keys.append(key)
            print(f"{key}: {tensor1.shape}, {tensor2.shape}; norm: {torch.norm(tensor1)}, {torch.norm(tensor2)}")
            all_same = False

    if all_same:
        print("✅ All parameters are exactly the same.")
    else:
        print(f"Difference found in: {different_keys}")