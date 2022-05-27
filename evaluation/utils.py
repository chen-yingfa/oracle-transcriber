def get_param_count(module):
    cnt = 0
    for _, param in module.named_parameters():
        cnt += param.numel()
    return cnt