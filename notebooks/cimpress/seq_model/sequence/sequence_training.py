"""
    The main file of sequence model training.
    Here we just use dummy data for testing only.
"""
import torch as th
import argparse
from models.seq_model import sequence_model, seq_loss_fn


def build_dummy_data():
    steps = 6 + 1       # will use 6 step as input, but the last 6 as output groud truth
    fg_emb_dim = 576    # the first sector of input sequence, which is the foreground image embeddings
    mb_size = 4         # the number of samples

    seqs = th.ones([steps, mb_size, fg_emb_dim + 5])
    bk_embs = th.randn(mb_size, fg_emb_dim)
    cls_labels = th.randint(0,2, (mb_size,))

    return seqs, bk_embs, cls_labels

def main(args):
    # 0. Mis parts
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print("Use device {}".format(device))

    # 1. prepare dataset
    if args.data_set == 'dummy':
        in_seqs, bk_embs, cls_labels = build_dummy_data()
    elif args.data_set == 'raw':
        in_seqs, bk_embs, cls_labels = None, None, None
    else:
        raise Exception("Only support dummy dataset...!")
    print("Input sequence has shape {}".format(in_seqs.shape))
    print("Background embedding has shape {}".format(bk_embs.shape))
    print("Class labels has shape {}".format(cls_labels.shape))
    in_seqs = in_seqs.to(device)
    bk_embs = bk_embs.to(device)
    cls_labels = cls_labels.to(device)

    # 2. define models(s)
    seq_model = sequence_model(input_size=args.seq_in_dim,
                               input_hid_size=args.input_hid_size,
                               hidden_size=args.hid_dim,
                               num_layers=args.num_layers)
    seq_model = seq_model.to(device)

    # 3. define loss and optimizer
    # will direct use the customized loss function.
    optim = th.optim.Adam(seq_model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 4. training loop
    for epoch in range(args.epochs):
        # for our dummy dataset, no need of mini-batch mode
        seq_model.train()

        h_0 = th.stack([bk_embs for _ in range(args.num_layers)]).to(device)
        # print("Hidden 0 shape {}".format(h_0.shape))

        c_0 = th.zeros_like(h_0).to(device)

        out_seqs_logits, cls_logits = seq_model(in_seqs[:-1,], (h_0, c_0))

        loss = seq_loss_fn(out_seqs_logits, in_seqs[1:,], cls_logits, cls_labels, alpha=0.2)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print("In epoch: {:03d} | dummy loss: {:.6f}".format(epoch, loss.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sequence Generation Model")
    parser.add_argument("--data-set", type=str, default="dummy")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seq-in_dim", type=int, default=581)
    parser.add_argument("--input-hid_size", type=int, default=576)
    parser.add_argument("--hid-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()
    print(args)
    main(args)