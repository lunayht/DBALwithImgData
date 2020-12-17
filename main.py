import os
import time
import argparse
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from scipy.ndimage.filters import gaussian_filter1d

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

def save_as_npy(data: np.ndarray, folder: str = "result_npy", name: str):
    """Save result as npy file
    
    Attributes:
        data: np array to be saved as npy file,
        folder: result folder name,
        name: npy filename
    """
    file_name = os.path.join(folder, name+".npy")
    np.save(file_name, data)
    print(f"Saved: {file_name}")

def plot_results(data: dict):
    """Plot results histogram using matplotlib"""
    sns.set()
    for key in data.keys():
        # data[key] = gaussian_filter1d(data[key], sigma=0.9) # for smoother graph
        plt.plot(data[key], label=key)
    plt.show()


def print_elapsed_time(start_time: float, exp: int, acq_func: str):
    """Print elapsed time for each experiment of acquiring

    Attributes:
        start_time: Starting time (in time.time()),
        exp: Experiment iteration
        acq_func: Name of acquisition function
    """
    elp = time.time() - start_time
    print(
        f"********** Experiment {exp} ({acq_func}): {int(elp//3600)}:{int(elp%3600//60)}:{int(elp%60)} **********"
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
    if args.determ:
        state_loop = [True, False] # dropout VS non-dropout
    else:
        state_loop = [True] # run dropout only
    
    for state in state_loop:
        for i, acq_func in enumerate(acq_functions):
            avg_hist = []
            test_scores = []
            acq_func_name = str(acq_func).split(" ")[1] + "-MC_dropout=" + str(state)
            print(f"\n---------- Start {acq_func_name} training! ----------")
            for e in range(args.experiments):
                start_time = time.time()
                estimator = load_CNN_model(args, device)
                print(
                    f"********** Experiment Iterations: {e+1}/{args.experiments} **********"
                )
                training_hist, test_score = active_learning_procedure(
                    query_strategy=acq_func,
                    X_val=datasets["X_val"],
                    y_val=datasets["y_val"],
                    X_test=datasets["X_test"],
                    y_test=datasets["y_test"],
                    X_pool=datasets["X_pool"],
                    y_pool=datasets["y_pool"],
                    X_init=datasets["X_init"],
                    y_init=datasets["y_init"],
                    estimator=estimator,
                    T=args.dropout_iter,
                    n_query=args.query,
                    training=state
                )
                avg_hist.append(training_hist)
                test_scores.append(test_score)
                print_elapsed_time(start_time, e + 1, acq_func_name)
            avg_hist = np.average(np.array(avg_hist), axis=0)
            avg_test = sum(test_scores) / len(test_scores)
            print(f"Average Test score for {acq_func_name}: {avg_test}")
            results[acq_func_name] = avg_hist
            save_as_npy(data=avg_hist, folder=args.result_dir ,name=acq_func_name)
    print("--------------- Done Training! ---------------")
    return results


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
                            3-bald, 4-var_ratios, 5-mean_std (default: 0)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=100,
        metavar="V",
        help="validation set size (default: 100)",
    )
    parser.add_argument(
        "--determ",
        action="store_true",
        help="Compare with deterministic models (default: False)",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="result_npy",
        metavar="SD",
        help="Save npy file in this folder (default: result_npy)",
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

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    results = train_active_learning(args, device, datasets)
    plot_results(data=results)


if __name__ == "__main__":
    main()
