import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys, os, time, math
sys.path.append("../../")
from utils import *
from model import RT

import warnings
warnings.filterwarnings("ignore")   # Suppress the RunTimeWarning on unicode


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--preprocess', action='store_true')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.15)
parser.add_argument('--clip', type=float, default=0.15)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--ksize', type=int, default=7)
parser.add_argument('--n_level', type=int, default=3)
parser.add_argument('--log-interval', type=int, default=100, metavar='N')
parser.add_argument('--lr', type=float, default=2)
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--rnn_type', type=str, default='GRU')
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--h', type=int, default=8)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--validseqlen', type=int, default=320)
parser.add_argument('--seq_len', type=int, default=400)
parser.add_argument('--dataset', type=str, default='ptb')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda")

base_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_path,'data/{}/'.format(args.dataset))
s_dir = os.path.join(base_path,'output/')


if args.preprocess:   # if the data need to preprocess
    file, file_len, valfile, valfile_len, testfile, testfile_len, corpus = data_generator(args, data_dir)
    n_characters = len(corpus.dict)
    train_data = batchify(char_tensor(corpus, file), args.batch_size)
    torch.save(train_data, data_dir + 'train_data.pt')
    val_data = batchify(char_tensor(corpus, valfile), 1)
    torch.save(val_data, data_dir + 'val_data.pt')
    test_data = batchify(char_tensor(corpus, testfile), 1)
    torch.save(test_data, data_dir + 'test_data.pt')
    print("Corpus size: ", n_characters)
else:
    train_data = torch.load(data_dir + 'train_data.pt')
    val_data = torch.load(data_dir + 'val_data.pt')
    test_data = torch.load(data_dir + 'test_data.pt')
n_characters = 49
train_data.to(device)
val_data.to(device)
test_data.to(device)
print (args)

dropout = args.dropout
emb_dropout = dropout
model = RT(args.d_model, n_characters, h=args.h, n=args.n, rnn_type=args.rnn_type,
            ksize=args.ksize, n_level=args.n_level, dropout=dropout, emb_dropout=emb_dropout)
model_name = "data_{}_d_{}_h_{}_type_{}_ksize_{}_level_{}_n_{}_lr_{}_dropout_{}".format(
                args.dataset, args.d_model, args.h, args.rnn_type, args.ksize, 
                 args.n_level, args.n, args.lr, args.dropout)

message_filename = s_dir + 'r_' + model_name + '.txt'
model_filename = s_dir + 'm_' + model_name + '.pt'
with open(message_filename, 'w') as out:
    out.write('start\n')

model.to(device)


criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate(source):
    model.eval()
    total_loss = 0
    count = 0
    source_len = source.size(1)
    with torch.no_grad():
        for batch, i in enumerate(range(0, source_len - 1, args.validseqlen)):
            if i + args.seq_len - args.validseqlen >= source_len:
                continue
            inp, target = get_batch(source, i, args)    
            output = model(inp)
            eff_history = args.seq_len - args.validseqlen
            final_output = output[:, eff_history:].contiguous().view(-1, n_characters)
            final_target = target[:, eff_history:].contiguous().view(-1)
            loss = criterion(final_output, final_target)

            total_loss += loss.data * final_output.size(0)
            count += final_output.size(0)

        val_loss = total_loss.item() / count * 1.0
        return val_loss


def train(epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    losses = []
    source = train_data
    source_len = source.size(1)
    for batch_idx, i in enumerate(range(0, source_len - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= source_len:
            continue
        inp, target = get_batch(source, i, args)
        optimizer.zero_grad()
        output = model(inp)
        eff_history = args.seq_len - args.validseqlen
        final_output = output[:, eff_history:].contiguous().view(-1, n_characters)
        final_target = target[:, eff_history:].contiguous().view(-1)
        loss = criterion(final_output, final_target)
        loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            losses.append(cur_loss)
            elapsed = time.time() - start_time
            message = ('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                  'loss {:5.3f} | bpc {:5.3f}'.format(
                epoch, batch_idx, int((source_len-0.5) / args.validseqlen), lr,
                elapsed * 1000 / args.log_interval, cur_loss, cur_loss / math.log(2)))
            print (message)
            output_s(message, message_filename)
            total_loss = 0
            start_time = time.time()

    return sum(losses) * 1.0 / len(losses)


def main():
    global lr
    try:
        print("Training for %d epochs..." % args.epochs)
        all_losses = []
        best_vloss = 1e7
        for epoch in range(1, args.epochs + 1):
            loss = train(epoch)

            vloss = evaluate(val_data)
            message = '-' * 89 + '\n' + '| End of epoch {:3d} | valid loss {:5.3f} | valid bpc {:8.3f}'.format(
                epoch, vloss, vloss / math.log(2))
            print (message)
            output_s(message, message_filename)

            test_loss = evaluate(test_data)
            message = '=' * 89 + '\n' + '| End of epoch {:3d} | test loss {:5.3f} | test bpc {:8.3f}'.format(
                epoch, test_loss, test_loss / math.log(2)) + '\n' + '=' * 89
            print (message)
            output_s(message, message_filename)

            if epoch > 5 and vloss > max(all_losses[-3:]):
                lr = lr / 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_losses.append(vloss)

            if vloss < best_vloss:
                print("Saving...")
                save(model, model_filename)
                best_vloss = vloss

    except KeyboardInterrupt:
        print('-' * 89)
        print("Saving before quit...")
        save(model, model_filename)
    

    # Run on test data.
    test_loss = evaluate(test_data)
    message = '=' * 89 + '\n' + '| End of training | test loss {:5.3f} | test bpc {:8.3f}'.format(
        test_loss, test_loss / math.log(2)) + '\n' + '=' * 89
    print (message)
    output_s(message, message_filename)

# train_by_random_chunk()
if __name__ == "__main__":
    main()

