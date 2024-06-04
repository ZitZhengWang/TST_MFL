import torch


models = {}
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

# make函数加载并初始化模型，参数都放在 **kwargs中，需要熟悉对应的模型；
def make(name, **kwargs):
    if name is None:
        return None
    model = models[name](**kwargs)
    if torch.cuda.is_available():
        model.cuda()
    return model


def load(model_sv, name=None):
    if name is None:
        name = 'model'
    model = make(model_sv[name], **model_sv[name + '_args'])    # 建立一个空的 待加载的模型；即train_classifier训练出来的模型；
    model.load_state_dict(model_sv[name + '_sd'])    # 加载该模型的参数
    return model

