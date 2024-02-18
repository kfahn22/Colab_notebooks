# Things to try for CUDA out of memory error:


pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

pipe.enable_sequential_cpu_offload()

pipe.enable_attention_slicing()

gc.collect()
torch.cuda.empty_cache()