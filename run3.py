'''

Script for running many different experiments

'''

import os

type = 'binary'
types = ['dilation', 'erosion', 'binary', 'cannyedge']
types = f'{types[0]} {types[1]} {types[2]} {types[3]}'
tuning_ps = [4,9,9,6]
tuning_ps = f'{tuning_ps[0]} {tuning_ps[1]} {tuning_ps[2]} {tuning_ps[3]}'
learning_rates=[0.0001, 0.00001, 0.001, 0.000001, 0.01]
start_epoch = 0
max_epoch = 10000
cuda = 3
cudas = f'{cuda} {cuda} {cuda} {cuda}'


'''Fine-tune cascaded ctunet'''
for lr in learning_rates:
    print(f'python train_cascaded_ctunet.py -start_epoch {start_epoch} -max_epoch {max_epoch} -cudas {cudas} -lr {lr} -types {types} -tuning_ps {tuning_ps}')
    os.system(f'python train_cascaded_ctunet.py -start_epoch {start_epoch} -max_epoch {max_epoch} -cudas {cudas} -lr {lr} -types {types} -tuning_ps {tuning_ps}')

'''Run Train cascadable tunet'''
# type = types[0]
# for lr in learning_rates:
#     print(f'python train_ctunet.py -type {type} -max_epoch {max_epoch} -cuda {cuda} -lr {lr} -no_sigmoid -no_bias -ds_length 0.1')
#     os.system(f'python train_ctunet.py -type {type} -max_epoch {max_epoch} -cuda {cuda} -lr {lr} -no_sigmoid -no_bias -ds_length 0.1')


'''Test all possible cascaded experiments beginning with type1'''
# type1 = type
# for type2 in types:
#     if type2 == type1: continue
#     for type3 in types:
#         if type3 == type2 or type3 ==type1: continue
#         for type4 in types:
#             if type4 == type3 or type4 == type2 or type4 ==type1: continue
#             print(f'python test_cascaded_ctunet.py -types {type1} {type2} {type3} {type4}  -cudas {cudas}')
#             os.system(f'python test_cascaded_ctunet.py -types {type1} {type2} {type3} {type4}  -cudas {cudas}')

'''Run all possible cascaded experiments'''
# for lr in learning_rates:
#     # for type1 in types:
#     type1='binary'
#     for type2 in types:
#         if type2 == type1: continue
#         for type3 in types:
#             if type3 == type2 or type3 ==type1: continue
#             for type4 in types:
#                 if type4 == type3 or type4 == type2 or type4 ==type1: continue
#                 print(f'python train_cascaded_ctunet.py -types {type1} {type2} {type3} {type4} -max_epoch {max_epoch} -cudas {cudas} -lr {lr}')
#                 os.system(f'python train_cascaded_ctunet.py -types {type1} {type2} {type3} {type4} -max_epoch {max_epoch} -cudas {cudas} -lr {lr}')