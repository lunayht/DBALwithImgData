import argparse
import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

from load_data import LoadData
from cnn_model import ConvNN
from active_learning import select_acq_function, active_learning_procedure

def load_CNN_model(args, device):
    """Load new model each time for different acqusition function"""
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
    results["testscore"] = np.zeros((len(acq_functions),1), dtype=float)
    for i, acq_func in enumerate(acq_functions):
        avg_hist = []
        test_scores = []
        acq_func_name = str(acq_func).split(" ")[1]
        estimator = load_CNN_model(args, device)
        print(f"\n---------- Start {acq_func_name} training! ----------")
        for e in range(args.experiments):
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
        avg_hist = np.average(np.array(avg_hist), axis=0)
        avg_test = np.average(np.array(test_scores), axis=0)
        results[acq_func_name] = avg_hist
        results["testscore"][i] = avg_test
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
                            3-bald (default: 0)",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = dict()
    DataLoader = LoadData()
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
    print(f"Average test score for each acquisition function are: {results["testscore"]}")

if __name__ == "__main__":
    main()
