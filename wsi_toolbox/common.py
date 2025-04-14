import timm


# DEFAULT_MODEL = 'uni'
# DEFAULT_MODEL_NAME = 'UNI'
# EMBEDDING_SIZE = 1024
DEFAULT_MODEL = 'gigapath'
DEFAULT_MODEL_NAME = 'GigaPath'
EMBEDDING_SIZE = 1536


def create_model(model_name):
    if model_name == 'uni':
        return timm.create_model('hf-hub:MahmoodLab/uni',
                                 pretrained=True,
                                 dynamic_img_size=True,
                                 init_values=1e-5)

    if model_name == 'gigapath':
        return timm.create_model('hf_hub:prov-gigapath/prov-gigapath',
                                 pretrained=True,
                                 dynamic_img_size=True)

    raise ValueError('Invalid model_name', model_name)

