from functools import wraps

def parametrize_shapes(shape_list, batch_size_range):
    """
    For each shape config, sweep batch_size_range and call
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for shape in shape_list:
                for bs in batch_size_range:
                    # unpack the shape parameters
                    if len(shape) == 3:
                        num_frames, H, W = shape
                    else:
                        raise ValueError("shape must be (num_frames, H, W)")
                    print(f"\n==== case: bs={bs}, num_frames={num_frames}, H={H}, W={W}, desc= ====")
                    result = func(
                        bs=bs, num_frames=num_frames, H=H, W=W,  **kwargs
                    )
                    results.append((bs, num_frames, H, W, result))
            return results
        return wrapper
    return decorator
