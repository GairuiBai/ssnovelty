"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options

from lib.model import SSnovelty

##
# def main():
""" Training
"""

##
# ARGUMENTS
opt = Options().parse()

##
# LOAD DATA
# dataloader = load_data(opt)

##
# LOAD MODEL

for i in range(10):
    f = open('./output/testclass.txt', 'a', encoding='utf-8-sig', newline='\n')
    f.write('class:' + str(i) + '\n')
    f.close()

    opt.normalclass = i
    model = SSnovelty(opt)
    model.train()

# if __name__ == '__main__':
#     main()
