'''
Modified version for full net lora
(Lora for ResBlock and up/down sample block)
'''
import os, sys
import re
import torch

from modules import shared, devices, sd_models
import lora
from locon_compvis import LoConModule, LoConNetworkCompvis, create_network_and_apply_compvis


try:
    '''
    Hijack Additional Network extension
    '''
    now_dir = os.path.dirname(os.path.abspath(__file__))
    addnet_path = os.path.join(now_dir, '..', '..', 'sd-webui-additional-networks/scripts')
    sys.path.append(addnet_path)
    import lora_compvis
    import scripts
    scripts.lora_compvis = lora_compvis
    scripts.lora_compvis.LoRAModule = LoConModule
    scripts.lora_compvis.LoRANetworkCompvis = LoConNetworkCompvis
    scripts.lora_compvis.create_network_and_apply_compvis = create_network_and_apply_compvis
    print('LoCon Extension hijack addnet extension successfully')
except:
    print('Additional Network extension not installed, Only hijack built-in lora')


'''
Hijack sd-webui LoRA
'''
re_digits = re.compile(r"\d+")

re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")

re_unet_down_blocks_res = re.compile(r"lora_unet_down_blocks_(\d+)_resnets_(\d+)_(.+)")
re_unet_mid_blocks_res = re.compile(r"lora_unet_mid_block_resnets_(\d+)_(.+)")
re_unet_up_blocks_res = re.compile(r"lora_unet_up_blocks_(\d+)_resnets_(\d+)_(.+)")

re_unet_downsample = re.compile(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv(.+)")
re_unet_upsample = re.compile(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv(.+)")

re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")


def convert_diffusers_name_to_compvis(key):
    def match(match_list, regex):
        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, re_unet_down_blocks):
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_mid_blocks):
        return f"diffusion_model_middle_block_1_{m[1]}"

    if match(m, re_unet_up_blocks):
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_down_blocks_res):
        block = f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_mid_blocks_res):
        block = f"diffusion_model_middle_block_{m[0]*2}_"
        if m[1].startswith('conv1'):
            return f"{block}in_layers_2{m[1][len('conv1'):]}"
        elif m[1].startswith('conv2'):
            return f"{block}out_layers_3{m[1][len('conv2'):]}"
        elif m[1].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[1][len('time_emb_proj'):]}"
        elif m[1].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[1][len('conv_shortcut'):]}"

    if match(m, re_unet_up_blocks_res):
        block = f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_downsample):
        return f"diffusion_model_input_blocks_{m[0]*3+3}_0_op{m[1]}"

    if match(m, re_unet_upsample):
        return f"diffusion_model_output_blocks_{m[0]*3 + 2}_{1+(m[0]!=0)}_conv{m[1]}"

    if match(m, re_text_block):
        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key


class LoraOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename


class LoraModule:
    def __init__(self, name):
        self.name = name
        self.multiplier = 1.0
        self.modules = {}
        self.mtime = None


class LoraUpDownModule:
    def __init__(self):
        self.up = None
        self.down = None
        self.alpha = None


def load_lora(name, filename):
    lora = LoraModule(name)
    lora.mtime = os.path.getmtime(filename)

    sd = sd_models.read_state_dict(filename)

    keys_failed_to_match = []

    for key_diffusers, weight in sd.items():
        fullkey = convert_diffusers_name_to_compvis(key_diffusers)
        key, lora_key = fullkey.split(".", 1)
        
        sd_module = shared.sd_model.lora_layer_mapping.get(key, None)
        if sd_module is None:
            keys_failed_to_match.append(key_diffusers)
            continue

        lora_module = lora.modules.get(key, None)
        if lora_module is None:
            lora_module = LoraUpDownModule()
            lora.modules[key] = lora_module

        if lora_key == "alpha":
            lora_module.alpha = weight.item()
            continue

        if type(sd_module) == torch.nn.Linear:
            weight = weight.reshape(weight.shape[0], -1)
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.Conv2d:
            if lora_key == "lora_down.weight":
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], sd_module.kernel_size, sd_module.stride, sd_module.padding, bias=False)
            elif lora_key == "lora_up.weight":
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'
        
        with torch.no_grad():
            module.weight.copy_(weight)

        module.to(device=devices.device, dtype=devices.dtype)

        if lora_key == "lora_up.weight":
            lora_module.up = module
        elif lora_key == "lora_down.weight":
            lora_module.down = module
        else:
            assert False, f'Bad Lora layer name: {key_diffusers} - must end in lora_up.weight, lora_down.weight or alpha'

    # print(shared.sd_model.lora_layer_mapping)

    if len(keys_failed_to_match) > 0:
        print(f"Failed to match keys when loading Lora {filename}: {keys_failed_to_match}")
    
    import json
    print(json.dumps({k:str(v) for k,v in shared.sd_model.lora_layer_mapping.items()}, indent=2))
    return lora


lora.convert_diffusers_name_to_compvis = convert_diffusers_name_to_compvis
lora.load_lora = load_lora
print('LoCon Extension hijack built-in lora successfully')