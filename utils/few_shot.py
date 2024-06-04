import torch
def split_shot_query(data, way, shot, query, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape)    # [batch, way, s+q, channal, H, W] or [batch, way, s+q, patch_num, C, H, W]
    x_shot, x_query = data.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()    # [3,5,15,3,84,84]
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape) # [3,75,3,84,84] or [3, 75, 36, 3, 21, 21]
    return x_shot, x_query

def make_nk_label(n, k, ep_per_batch=1):
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label

def make_nkp_label(n, k, patch_num, ep_per_batch=1):
    # l_query: torch.Size([3, 75, 36, 3, 21, 21])
    # label要和 query对应
    label = torch.arange(n).unsqueeze(1).expand(n, k*patch_num).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label

