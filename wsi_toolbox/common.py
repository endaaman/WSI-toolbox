import os
import torch
import timm
from timm.layers import SwiGLUPacked


DEFAULT_BACKEND = 'tqdm'

DEFAULT_MODEL = 'uni'
DEFAULT_MODEL_LABEL = 'UNI'
EMBEDDING_SIZE = 1024

# DEFAULT_MODEL = 'gigapath'
# DEFAULT_MODEL_LABEL = 'GigaPath'
# EMBEDDING_SIZE = 1536

# DEFAULT_MODEL = 'virchow2'
# DEFAULT_MODEL_LABEL = 'Virchow2'
# EMBEDDING_SIZE = 1280

# DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gigapath')
# DEFAULT_MODEL_LABEL = os.getenv('DEFAULT_MODEL_LABEL', 'GigaPath')

print(f'DEFAULT_MODEL {DEFAULT_MODEL} ')
print(f'DEFAULT_MODEL_LABEL {DEFAULT_MODEL_LABEL}')

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

    if model_name == 'virchow2':
        return timm.create_model("hf-hub:paige-ai/Virchow2",
                                  pretrained=True,
                                  mlp_layer=SwiGLUPacked,
                                  act_layer=torch.nn.SiLU)

    raise ValueError('Invalid model_name', model_name)

