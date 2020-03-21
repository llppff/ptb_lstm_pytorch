#! /bin/bash  
#!/usr/bin/env python
adabound：python main.py --cuda --embedding_size 1500 --num_hid_unit 1500 --epochs 200 --tied --optim adabound
adabound_gradient_noise：python main.py --cuda --embedding_size 1500 --num_hid_unit 1500 --epochs 200 --tied --optim adabound --gamma 0.55 --ita 0.01
