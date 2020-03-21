# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import logging
import data
import model
import torch.optim as optim
from adabound import AdaBound

def create_optimizer(args, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        return optim.Adagrad(model_params, args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'amsgrad':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True)
    elif args.optim == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    else:
        assert args.optim == 'amsbound'
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay, amsbound=True)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz#//代表整除
    #data.narrow(a,b,c),a代表对行还是列操作，0为行，1为列，b代表开始的位置，c代表取得个数（取几行/列）
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()#在调用contiguous()之后，PyTorch会开辟一块新的内存空间存放变换之后的数据,而view是新数据与原数据共享一块内存
    return data.to(device)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def train():
    model.train()
    total_loss = 0.
    total_perplexity = 0
    # start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch = 0
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)

        optimizer = create_optimizer(args, model.parameters())
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        print("loss:" + str(loss.item()))
        total_loss += loss.item()
        optimizer.step()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

    train_loss = total_loss / batch
    print("train_loss" + str(train_loss))
    # train_perplexity = total_perplexity / batch
    train_perplexity=0
    #     elapsed = time.time() - start_time
    #     # logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
    #     #         'loss {:5.2f} | ppl {:8.2f}'.format(
    #     #     epoch, batch, len(train_data) // args.bptt, lr,
    #     #     elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
    #     print("train_loss:" + str(train_loss))
    logging.info('train_perplexity:{:5.2f}'.format(train_perplexity))
    total_loss = 0
    test_perplexity = 0
        #     start_time = time.time()

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def initLogging(logFilename):
    # logging.basicConfig()函数是一个一次性的简单配置工具，即只有在第一次调用该函数时会起作用，后续再次调用该函数时完全不会产生任何操作
    #函数调整日志级别、输出格式等
    logging.basicConfig(
        #设置日志器级别
        level = logging.DEBUG,
        #指定日志格式字符串，即指定日志输出时所包含的字段信息以及它们的顺序。logging模块定义的格式字段下面会列出
        # format='%(asctime)s-%(levelname)s-%(message)s',
        format='%(message)s',
        #指定日期/时间格式
        # datefmt  = '%y-%m-%d %H:%M',
        #指定日志输出目标文件的文件名，指定该设置项后日志信息就不会被输出到控制台了
        filename = logFilename,
        #定日志文件的打开模式，默认为'a'，该选项要在filename指定时才有效
        filemode = 'w');
    #日志器（logger）是入口，真正干活儿的是处理器（handler），处理器（handler）还可以通过过滤器（filter）和格式器（formatter）对要输出的日志内容做过滤和格式化等处理操作
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(levelname)s-%(message)s')
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# 建立解析对象
parser = argparse.ArgumentParser(description='LSTM PTB Language Model')
#增加属性
parser.add_argument('--data', type=str, default='ptb_lstm_pytorch/data/ptb',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--embedding_size', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--num_hid_unit', type=int, default=1500,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=22,
                    help='initial learning rate')
# parser.add_argument('--clip', type=float, default=0.25,
#                     help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=40, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--seed_gpu', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')

parser.add_argument('--optim', default='sgd', type=str, help='optimizer',
                    choices=['sgd', 'adagrad', 'adam', 'amsgrad', 'adabound', 'amsbound'])
parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
parser.add_argument('--final_lr', default=0.1, type=float,
                    help='final learning rate of AdaBound')
parser.add_argument('--gamma', default=1e-3, type=float,)
parser.add_argument('--ita',default=1e-2, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='weight decay for optimizers')

# 属性给与args实例： 把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中，使用即可。
args = parser.parse_args()

torch.manual_seed(args.seed)#为CPU设置种子用于生成随机数，以使得生成随机数的结果是确定的
torch.cuda.manual_seed(args.seed_gpu)#为当前GPU设置随机种子

device = torch.device("cuda" if args.cuda else "cpu")

#将数据集设置好，见data.py中
corpus = data.Corpus(args.data)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.embedding_size, args.num_hid_unit, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()

initLogging('test.log')

lr = args.lr
# best_val_loss = None

try:
    for epoch in range(1, args.epochs+1):
        logging.info('epoch:{0}'.format(epoch))
        # epoch_start_time = time.time()
        train()
        test_loss = evaluate(test_data)
        # test_perplexity = 2 ** test_loss
        # logging.info('test_perolexity {:5.2f}'.format(test_perplexity))
        # logging.info('-' * 89)
        # logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #         'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                    val_loss, math.exp(val_loss)))
        # print("valid_loss:" + str(val_loss))
        val_loss = evaluate(val_data)
        # val_perplexity = 2 ** val_loss
        # logging.info('valid_perplexity:{:5.2f}'.format(val_perplexity))
        # logging.info('-' * 89)
        # if not best_val_loss or val_loss < best_val_loss:
        #     with open(args.save, 'wb') as f:
        #         torch.save(model, f)
        #     best_val_loss = val_loss
        # else:
        #     lr /= 2.5
except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')

# with open(args.save, 'rb') as f:
#     model = torch.load(f)
#
#     model.rnn.flatten_parameters()


# print("test_loss:" + str(test_loss))
# logging.info('=' * 89)
# logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))

# logging.info('=' * 89)


