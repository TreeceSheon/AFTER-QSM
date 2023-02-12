import torch

state = torch.load('checkpoints/UnetY_episode13_b5_150.pkl')['model2_state']
state1 = torch.load('checkpoints/UnetY_episode13_b5_150.pkl')['model1_state']

names = state.keys()
module = state.values()
new_names = list(names)

for idx, new_name in enumerate(new_names):

    new_names[idx] = new_name.replace('sfe', 'feature_extraction')
    new_name = new_names[idx]

    new_names[idx] = new_name.replace('rdbs', 'blocks')
    new_name = new_names[idx]

    new_names[idx] = new_name.replace('output', 'conv')
    new_name = new_names[idx]

    new_names[idx] = new_name.replace('last_conv', 'output')
    new_name = new_names[idx]

    new_names[idx] = new_name.replace('gff', 'feature_fusion_global')
    new_name = new_names[idx]

    new_names[idx] = new_name.replace('lff', 'feature_fusion_local')
    new_name = new_names[idx]
    new_names[idx] = new_name.replace('CNN', 'layers')
    new_name = new_names[idx]

new_state = dict(zip(new_names, module))

from networks.Refinement import Refinement
import torch.nn as nn

rdn = nn.DataParallel(Refinement())

torch.save({'refine_model_state': new_state, 'recon_model_state': state1}, 'AFTER-QSM.pkl')
