import argparse

import numpy as np
import torch
from sklearn.metrics import r2_score

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.evaluation import get_openml_predictions, TOY_TASKS_REGRESSION
from tfmplayground.mdn import MDNDecoder, MDNCriterion
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import TabICLPriorDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed

parser = argparse.ArgumentParser()

parser.add_argument(
    "--saveweights",
    type=str,
    default="nanotabpfn_weights.pth",
    help="path to save the trained model to",
)
parser.add_argument("--heads", type=int, default=6, help="number of attention heads")
parser.add_argument(
    "--embeddingsize",
    type=int,
    default=192,
    help="the size of the embeddings used for the cells",
)
parser.add_argument(
    "--hiddensize", type=int, default=768, help="size of the hidden layer of the mlps"
)
parser.add_argument(
    "--layers", type=int, default=6, help="number of transformer layers"
)
parser.add_argument(
    "--num_components", type=int, default=8, help="number of GMM components in the MDN decoder"
)
parser.add_argument(
    "--batchsize",
    type=int,
    default=1,
    help="batch size used during training (before gradient accumulation)",
)
parser.add_argument(
    "--accumulate",
    type=int,
    default=1,
    help="number of gradients to accumulate before updating the weights",
)
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument(
    "--steps",
    type=int,
    default=100,
    help="number of steps that constitute one epoch (important for lr scheduler)",
)
parser.add_argument(
    "--epochs", type=int, default=10000, help="number of epochs to train for"
)
parser.add_argument(
    "--loadcheckpoint",
    type=str,
    default=None,
    help="checkpoint from which to continue training",
)
parser.add_argument(
    "--num_datapoints_min", type=int, default=32, help="minimum number of datapoints per dataset"
)
parser.add_argument(
    "--num_datapoints_max", type=int, default=128, help="maximum number of datapoints per dataset"
)
parser.add_argument(
    "--min_features", type=int, default=1, help="minimum number of features"
)
parser.add_argument(
    "--max_features", type=int, default=50, help="maximum number of features"
)
parser.add_argument(
    "--max_num_classes", type=int, default=10, help="maximum number of classes (used by TabICL prior)"
)
parser.add_argument(
    "--prior_type",
    type=str,
    default="mix_scm",
    choices=["mlp_scm", "tree_scm", "mix_scm", "dummy"],
    help="TabICL prior type",
)

args = parser.parse_args()

set_randomness_seed(2402)

device = get_default_device()
ckpt = None
if args.loadcheckpoint:
    ckpt = torch.load(args.loadcheckpoint)

prior = TabICLPriorDataLoader(
    num_steps=args.steps,
    batch_size=args.batchsize,
    num_datapoints_min=args.num_datapoints_min,
    num_datapoints_max=args.num_datapoints_max,
    min_features=args.min_features,
    max_features=args.max_features,
    max_num_classes=args.max_num_classes,
    prior_type=args.prior_type,
    device=device,
)

model = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=1,  # placeholder — decoder is replaced below
)
model.decoder = MDNDecoder(args.embeddingsize, args.hiddensize, args.num_components)

if ckpt:
    model.load_state_dict(ckpt["model"])

criterion = MDNCriterion()


class EvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks):
        self.tasks = tasks

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        model.eval()

        class MDNRegressor:
            """Minimal sklearn-compatible wrapper for evaluation."""
            def __init__(self, model, device):
                self.model = model
                self.device = device

            def fit(self, X_train, y_train):
                from tfmplayground.interface import get_feature_preprocessor
                self.preprocessor = get_feature_preprocessor(X_train)
                self.X_train = self.preprocessor.fit_transform(X_train)
                self.y_mean = float(np.mean(y_train))
                self.y_std = float(np.std(y_train, ddof=1)) + 1e-8
                self.y_train_n = (y_train - self.y_mean) / self.y_std

            def predict(self, X_test):
                X = np.concatenate((self.X_train, self.preprocessor.transform(X_test)))
                X_t = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
                y_t = torch.tensor(self.y_train_n, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    dist = self.model((X_t, y_t), single_eval_pos=len(self.X_train))
                preds_n = dist.mean.squeeze(0)
                return (preds_n * self.y_std + self.y_mean).cpu().numpy()

        regressor = MDNRegressor(model, device)
        predictions = get_openml_predictions(model=regressor, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, _) in predictions.items():
            scores.append(r2_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        print(
            f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg r2 score {avg_score:.3f}",
            flush=True,
        )


callbacks = [EvaluationLoggerCallback(TOY_TASKS_REGRESSION)]

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=criterion,
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=callbacks,
    ckpt=ckpt,
)

torch.save(trained_model.to("cpu").state_dict(), args.saveweights)
