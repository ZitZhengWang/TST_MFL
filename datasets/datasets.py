import os


# DEFAULT_ROOT = './materials'
DEFAULT_ROOT = '/home/zit/21class/ZhengWang/Datasets'

# /home/zit/21class/ZhengWang/Datasets/Caltech-UCSD Birds-200 2011/CUB_200_2011/images


datasets = {}
# 定义一个装饰器类，用它来在导入库的过程中，构建一个 name-数据集类 的字典；
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    # 如果 kwargs字典中没有指定数据集的根路径，则生成一个数据集的根路径
    if kwargs.get('root_path') is None:
        kwargs['root_path'] = os.path.join(DEFAULT_ROOT, name)

    # datasets是通过register注册好的一个保存 数据集名称-数据集类 的字典；
    # datasets[name] (**kwargs)调用 指定的数据集类，并且用 **kwargs中的参数来实例化一个数据集类
    dataset = datasets[name](**kwargs)
    return dataset

