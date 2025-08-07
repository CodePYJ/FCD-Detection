import importlib
import numpy as np

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def Normalize(img, num=1.):
    # img[img < 0.0] = 0.0
    img = (img - img.min())/(img.max() - img.min()) * num
    return img

def Normalize_res(img):
    img = img * 255.0
    return img

def dice_score(pred, targs, th=0.5):
    pred[pred>=th] = 1
    pred[pred<th] = 0
    targs[targs>0] = 1
    targs[targs<0] = 0
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def hist_mach(source, target):
    source = source.astype(int)
    target = target.astype(int)
    # 计算源图像和目标图像的直方图
    source_hist, bins1 = np.histogram(source, bins=256, range=[0, 256])
    target_hist, bins2 = np.histogram(target, bins=256, range=[0, 256])

    # 计算累积直方图
    source_cdf = source_hist.cumsum()
    source_cdf = (source_cdf / source_cdf[-1]).astype(np.float32)

    target_cdf = target_hist.cumsum()

    target_cdf = (target_cdf / target_cdf[-1]).astype(np.float32)

    # 使用累积直方图进行直方图匹配
    matched_cdf = np.interp(source_cdf, target_cdf, range(256)).astype(np.uint8)
    matched_img = matched_cdf[source]
    return matched_img