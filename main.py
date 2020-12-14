import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier

from load_data import LoadData
from cnn_model import ConvNN
from active_learning import select_acq_function, active_learning_procedure


def load_CNN_model(args, device):
    """Load new model each time for different acqusition function
    each experiments"""
    model = ConvNN().to(device)
    cnn_classifier = NeuralNetClassifier(
        module=model,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=device,
    )
    return cnn_classifier


def plot_results(data: dict):
    """Plot results histogram using matplotlib"""
    for key in data.keys():
        plt.plot(data[key], label=key)
        print(data[key])
    plt.show()


def print_elapsed_time(start_time: float, exp: int, acq_func: str):
    """Print elapsed time for each experiment of acquiring

    Attributes:
        start_time: Starting time (in time.time()),
        exp: Experiment iteration
        acq_func: Name of acquisition function
    """
    elp = start_time - time.time()
    print(
        f"********** Experiment {exp} ({acq_func}): {elp//3600:0.2d}:{elp%3600//60:0.2d}:{elp%60//1:0.2d} **********"
    )


def train_active_learning(args, device, datasets: dict) -> dict:
    """Start training process

    Attributes:
        args: Argparse input,
        estimator: Loaded model, e.g. CNN classifier,
        device: Cpu or gpu,
        datasets: Dataset dict that consists of all datasets,
    """
    acq_functions = select_acq_function(args.acq_func)
    results = dict()
    for i, acq_func in enumerate(acq_functions):
        avg_hist = []
        test_scores = []
        acq_func_name = str(acq_func).split(" ")[1]
        print(f"\n---------- Start {acq_func_name} training! ----------")
        for e in range(args.experiments):
            start_time = time.time()
            estimator = load_CNN_model(args, device)
            print(
                f"********** Experiment Iterations: {e+1}/{args.experiments} **********"
            )
            training_hist, test_score = active_learning_procedure(
                acq_func,
                datasets["X_val"],
                datasets["y_val"],
                datasets["X_test"],
                datasets["y_test"],
                datasets["X_pool"],
                datasets["y_pool"],
                datasets["X_init"],
                datasets["y_init"],
                estimator,
                args.dropout_iter,
                args.query,
            )
            avg_hist.append(training_hist)
            test_scores.append(test_score)
            print_elapsed_time(start_time, e + 1, acq_func_name)
        avg_hist = np.average(np.array(avg_hist), axis=0)
        avg_test = sum(test_scores) / len(test_scores)
        print(f"Average Test score for {acq_func_name}: {avg_test}")
        results[acq_func_name] = avg_hist.tolist()
    print("--------------- Done Training! ---------------")
    plot_results(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="EP",
        help="number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seed", type=int, default=369, metavar="S", help="random seed (default: 369)"
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=3,
        metavar="E",
        help="number of experiments (default: 3)",
    )
    parser.add_argument(
        "--dropout_iter",
        type=int,
        default=100,
        metavar="T",
        help="dropout iterations,T (default: 100)",
    )
    parser.add_argument(
        "--query",
        type=int,
        default=10,
        metavar="Q",
        help="number of query (default: 10)",
    )
    parser.add_argument(
        "--acq_func",
        type=int,
        default=0,
        metavar="AF",
        help="acqusition functions: 0-all, 1-uniform, 2-max_entropy, \
                            3-bald, 4-var_ratios (default: 0)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=100,
        metavar="V",
        help="validation set size (default: 100)",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = dict()
    DataLoader = LoadData(args.val_size)
    (
        datasets["X_init"],
        datasets["y_init"],
        datasets["X_val"],
        datasets["y_val"],
        datasets["X_pool"],
        datasets["y_pool"],
        datasets["X_test"],
        datasets["y_test"],
    ) = DataLoader.load_all()

    results = train_active_learning(args, device, datasets)


if __name__ == "__main__":
    main()
