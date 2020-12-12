import argparse
import torch
from load_data import LoadData
from cnn_model import ConvNN

def train(args, model, device, train_loader, optimizer, epoch):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, metavar="N",
                        help="input batch size for training (default: 128)")
    parser.add_argument("--epochs", type=int, default=50, metavar="EP",
                        help="number of epochs to train (default: 50)")
    parser.add_argument("--seed", type=int, default=369, metavar="S",
                        help="random seed (default: 369)")
    parser.add_argument("--experiments", type=int, default=3, metavar="E",
                        help="number of experiments (default: 3)")
    parser.add_argument("--acq_iter", type=int, default=98, metavar="AI",
                        help="acquisition iterations (default: 98)")
    parser.add_argument("--dropout_iter", type=int, default=100, metavar="DI",
                        help="dropout iterations (default: 100)")
    parser.add_argument("--query", type=int, default=10, metavar="Q",
                        help="number of query (default: 10)")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    DataLoader = LoadData()
    X_init, y_init, X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, \
        X_test, y_test = DataLoader.load_all()
    model = ConvNN().to(device)

if __name__ == "__main__":
    main()