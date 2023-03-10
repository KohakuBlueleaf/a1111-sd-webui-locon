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
        self.dim = None
    
    def inference(self, x):
        return self.up(self.down(x))


class LoraHadaModule:
    def __init__(self):
        self.w1a = None
        self.w1b = None
        self.w2a = None
        self.w2b = None
        self.alpha = None
        self.dim = None
        self.op = None
        self.extra_args = {}
        self.shape = None
    
    def inference(self, x):
        return self.op(
            x,
            ((self.w1a @ self.w1b) * (self.w2a @ self.w2b)).view(self.shape),
            **self.extra_args
        )


CON_KEY = {
    "lora_up.weight",
    "lora_down.weight"
}
HADA_KEY = {
    "hada_w1_a",
    "hada_w1_b",
    "hada_w2_a",
    "hada_w2_b",
}

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
        
        if lora_key in CON_KEY:
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

            module.to(device=devices.device, dtype=torch.float32)
            module.requires_grad_(False)

            if lora_key == "lora_up.weight":
                lora_module.up = module
            elif lora_key == "lora_down.weight":
                lora_module.down = module
                lora_module.dim = weight.shape[0]
        elif lora_key in HADA_KEY:
            if type(lora_module) != LoraHadaModule:
                alpha = lora_module.alpha
                lora_module = LoraHadaModule()
                lora_module.alpha = alpha
                lora.modules[key] = lora_module
            lora_module.shape = sd_module.weight.shape
            
            weight = weight.to(device=devices.device, dtype=torch.float32)
            weight.requires_grad_(False)
            
            if lora_key == 'hada_w1_a':
                lora_module.w1a = weight
            elif lora_key == 'hada_w1_b':
                lora_module.w1b = weight
                lora_module.dim = weight.shape[0]
            elif lora_key == 'hada_w2_a':
                lora_module.w2a = weight
            elif lora_key == 'hada_w2_b':
                lora_module.w2b = weight
            
            if type(sd_module) == torch.nn.Linear:
                lora_module.op = torch.nn.functional.linear
            elif type(sd_module) == torch.nn.Conv2d:
                lora_module.op = torch.nn.functional.conv2d
                lora_module.extra_args = {
                    'stride': sd_module.stride,
                    'padding': sd_module.padding
                }
            else:
                assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'
            
        else:
            assert False, f'Bad Lora layer name: {key_diffusers} - must end in lora_up.weight, lora_down.weight or alpha'

    if len(keys_failed_to_match) > 0:
        print(f"Failed to match keys when loading Lora {filename}: {keys_failed_to_match}")

    return lora


def lora_forward(module, input, res):
    if len(lora.loaded_loras) == 0:
        return res
    
    lora_layer_name = getattr(module, 'lora_layer_name', None)
    for lora_m in lora.loaded_loras:
        module = lora_m.modules.get(lora_layer_name, None)
        if module is not None and lora_m.multiplier:
            if isinstance(module, LoraUpDownModule):
                scale = lora_m.multiplier * (module.alpha / module.down.weight.size(0) if module.alpha else 1.0)
            else:
                scale = lora_m.multiplier * (module.alpha / module.dim if module.alpha else 1.0)
            
            if shared.opts.lora_apply_to_outputs and res.shape == input.shape:
                x = res
            else:
                x = input
            
            if hasattr(module, 'inference'):
                res = res + module.inference(x) * scale
            elif hasattr(module, 'up'):
                res = res + module.up(module.down(x)) * scale
            else:
                raise NotImplementedError(
                    "Your settings, extensions or models are not compatible with each other."
                )
    return res


lora.convert_diffusers_name_to_compvis = convert_diffusers_name_to_compvis
lora.load_lora = load_lora
lora.lora_forward = lora_forward
print('LoCon Extension hijack built-in lora successfully')