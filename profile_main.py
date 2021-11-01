from torch_lib.Nets import LargeNet as torchLNet, \
    MediumNet as torchMNet, SmallNet as torchSNet
from profile_lib.profile import profile

model = torchLNet()
num_ops, num_params = profile(model, [10, 3, 32, 32])
print('large', num_ops, num_params, sep='\t')
model = torchMNet()
num_ops, num_params = profile(model, [10, 3, 32, 32])
print('medium', num_ops, num_params, sep='\t')
model = torchSNet()
num_ops, num_params = profile(model, [10, 3, 32, 32])
print('small', num_ops, num_params, sep='\t')
