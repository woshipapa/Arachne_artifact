# # def get_mock_batch():
# #     rank = dist.get_rank()
# #     torch.cuda.set_device(rank)
# #     device = torch.device("cuda", rank)
# #     # images = torch.zeros((1, 221, 3, 1936, 1072), dtype=torch.float16, device=device)
# #     # batch = {
# #     #         "images": images,
# #     #         "prompt_embeds": torch.zeros((1, 124, 4096), dtype=torch.float16, device=device),
# #     #         "clip_text_embed": torch.zeros((1, 768), dtype=torch.float16, device=device),
# #     #         "first_ref_image": torch.zeros_like(images[:,:1])
# #     #     }
# #     if rank in [0, 1]:
# #         bs = 1
# #         # images = torch.zeros((bs, 189, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         images = torch.zeros((bs, 121, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         # images = torch.zeros((bs, 189, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((bs, 169, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((bs, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     elif rank in [2, 3]:
# #         # images = torch.zeros((1, 221, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         images = torch.zeros((1, 221, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((1, 124, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((1, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     elif rank in [4, 5]:
# #         bs = 1
# #         images = torch.zeros((bs, 97, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         # images = torch.zeros((bs, 157, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((bs, 120, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((bs, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     elif rank in [6, 7]:
# #         images = torch.zeros((1, 241, 3, 1072, 1936), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((1, 115, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((1, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     else:
# #         raise ValueError(f"Unsupported rank {rank}")

# #     return batch

# # def get_mock_batch_dp4sp8():
# #     global_rank = dist.get_rank()
# #     rank = global_rank % torch.cuda.device_count()
# #     torch.cuda.set_device(rank)
# #     device = torch.device("cuda", rank)

# #     if global_rank in [0, 1,2,3,4,5,6,7]:
# #         bs = 1
# #         images = torch.zeros((bs, 189, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         # images = torch.zeros((bs, 121, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         # images = torch.zeros((bs, 189, 3, 720, 1280), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((bs, 169, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((bs, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     elif global_rank in [8,9,10,11,12,13,14,15]:
# #         # images = torch.zeros((1, 221, 3, 720, 1280), dtype=torch.float16, device=device)
# #         images = torch.zeros((1, 221, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((1, 124, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((1, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     elif global_rank in [16,17,18,19,20,21,22,23]:
# #         bs = 1
# #         # images = torch.zeros((bs, 97, 3, 720, 1280), dtype=torch.float16, device=device)
# #         images = torch.zeros((bs, 97, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((bs, 120, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((bs, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     elif global_rank in [24,25,26,27,28,29,30,31]:
# #         # images = torch.zeros((1, 241, 3, 720, 1280), dtype=torch.float16, device=device)
# #         images = torch.zeros((1, 241, 3, 1936,1072), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((1, 115, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((1, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     else:
# #         raise ValueError(f"Unsupported rank {rank}")

# #     return batch

# # def get_mock_batch_sp12sp4_flex():
# #     global_rank = dist.get_rank()
# #     rank = global_rank % torch.cuda.device_count()
# #     torch.cuda.set_device(rank)
# #     device = torch.device("cuda", rank)
# #     group_ranks = None
    
# #     if global_rank in range(0,12):
# #         group_ranks = range(0,12)
# #         # images = torch.zeros((1, 241, 3, 720, 1280), dtype=torch.float16, device=device)
# #         images = torch.zeros((1, 241, 3, 1936,1072), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((1, 115, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((1, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     elif global_rank in range(12,16):
# #         group_ranks = range(12,16)
# #         images = torch.zeros((1, 25, 3, 720, 1280), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((1, 115, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((1, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }    
# #     group_names = f"seqlen{images.shape[1]}_gpus{len(group_ranks)}"
# #     mpu.register_custom_group(group_names, group_ranks)
# #     return batch    

# # def get_mock_batch_dp4sp8_flex():
# #     global_rank = dist.get_rank()
# #     rank = global_rank % torch.cuda.device_count()
# #     torch.cuda.set_device(rank)
# #     device = torch.device("cuda", rank)

# #     if global_rank in [0, 1,2,3]:
# #         bs = 1
# #         # images = torch.zeros((bs, 189, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         # images = torch.zeros((bs, 121, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         images = torch.zeros((bs, 189, 3, 720, 1280), dtype=torch.float16, device=device)
# #         group_ranks = [0, 1,2,3]
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((bs, 169, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((bs, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     elif global_rank in [8,9,10,11,12,13,14,15]:
# #         images = torch.zeros((1, 221, 3, 720, 1280), dtype=torch.float16, device=device)
# #         # images = torch.zeros((1, 221, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         group_ranks = [8,9,10,11,12,13,14,15]
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((1, 124, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((1, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     elif global_rank in [16,17,18,19]:
# #         bs = 1
# #         group_ranks = [16,17,18,19]
# #         # images = torch.zeros((bs, 97, 3, 720, 1280), dtype=torch.float16, device=device)
# #         images = torch.zeros((bs, 97, 3, 1936, 1072), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((bs, 120, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((bs, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     elif global_rank in [4,5,6,7,20,21,22,23,24,25,26,27,28,29,30,31]:
# #         group_ranks = [4,5,6,7,20,21,22,23,24,25,26,27,28,29,30,31]
# #         # images = torch.zeros((1, 241, 3, 720, 1280), dtype=torch.float16, device=device)
# #         images = torch.zeros((1, 241, 3, 1936,1072), dtype=torch.float16, device=device)
# #         batch = {
# #             "images": images,
# #             "prompt_embeds": torch.zeros((1, 115, 4096), dtype=torch.float16, device=device),
# #             "clip_text_embed": torch.zeros((1, 768), dtype=torch.float16, device=device),
# #             "first_ref_image": torch.zeros_like(images[:,:1])
# #         }
# #     else:
# #         raise ValueError(f"Unsupported rank {rank}")
    
# #     group_names = f"seqlen{images.shape[1]}_gpus{len(group_ranks)}"
# #     mpu.register_custom_group(group_names, group_ranks)
# #     return batch



# def get_mock_batch_flex4211_plus():
#     rank = dist.get_rank()
#     torch.cuda.set_device(rank)
#     device = torch.device("cuda", rank)
#     images = None
#     group_ranks = None
#     if rank in [6]:
#         bs = 1
#         group_ranks = [6]
#         images = torch.zeros((bs, 121, 3, 1936, 1072), dtype=torch.float16, device=device)
#         batch = {
#             "images": images,
#             "prompt_embeds": torch.zeros((bs, 112, 4096), dtype=torch.float16, device=device),
#             "clip_text_embed": torch.zeros((bs, 768), dtype=torch.float16, device=device),
#             "first_ref_image": torch.zeros_like(images[:,:1])
#         }
#     elif rank in [4, 5]:
#         images = torch.zeros((1, 221, 3, 1936, 1072), dtype=torch.float16, device=device)
#         group_ranks = [4, 5]
#         batch = {
#             "images": images,
#             "prompt_embeds": torch.zeros((1, 124, 4096), dtype=torch.float16, device=device),
#             "clip_text_embed": torch.zeros((1, 768), dtype=torch.float16, device=device),
#             "first_ref_image": torch.zeros_like(images[:,:1])
#         }
#     elif rank in [7]:
#         bs = 1
#         group_ranks = [7]
#         images = torch.zeros((bs, 97, 3, 1936, 1072), dtype=torch.float16, device=device)
#         batch = {
#             "images": images,
#             "prompt_embeds": torch.zeros((bs, 120, 4096), dtype=torch.float16, device=device),
#             "clip_text_embed": torch.zeros((bs, 768), dtype=torch.float16, device=device),
#             "first_ref_image": torch.zeros_like(images[:,:1])
#         }
#     elif rank in [0, 1, 2, 3]:
#         images = torch.zeros((1, 241, 3, 1072, 1936), dtype=torch.float16, device=device)
#         group_ranks = [0, 1, 2, 3]
#         batch = {
#             "images": images,
#             "prompt_embeds": torch.zeros((1, 115, 4096), dtype=torch.float16, device=device),
#             "clip_text_embed": torch.zeros((1, 768), dtype=torch.float16, device=device),
#             "first_ref_image": torch.zeros_like(images[:,:1])
#         }
#     else:
#         raise ValueError(f"Unsupported rank {rank}")
#     # group_names = f"seqlen{images.shape[1]}_gpus{len(group_ranks)}"
#     mpu.register_custom_group("VAE", vae_groups_plan)
#     mpu.register_custom_group("DIT", dit_groups_plan)
#     return batch





# def get_mock_batch_dp4cp2():
#     rank = dist.get_rank() % torch.cuda.device_count()
#     torch.cuda.set_device(rank)
#     device = torch.device("cuda", rank)

#     # 分组划分
#     if rank in range(0, 4):  # DP group 0
#         shape = (1, 241, 3, 1072, 1936)
#     elif rank in range(4, 8):  # DP group 1
#         shape = (1, 97, 3, 1072, 1936)
#     else:
#         raise ValueError(f"Unsupported rank {rank}")

#     # 通用构造逻辑
#     images = torch.zeros(shape, dtype=torch.float16, device=device)
#     batch = {
#         "images": images,
#         "prompt_embeds": torch.zeros((shape[0], 115, 4096), dtype=torch.float16, device=device),
#         "clip_text_embed": torch.zeros((shape[0], 768), dtype=torch.float16, device=device),
#         "first_ref_image": torch.zeros_like(images[:, :1])
#     }
#     return batch

# def get_batch_with_mock(batch):
#     fake = get_mock_batch()
#     # fake = get_mock_batch_sp12sp4_flex()
#     # fake = get_mock_batch_flex4211()
#     batch.update(fake)
#     return batch
