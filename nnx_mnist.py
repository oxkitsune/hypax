from flax import nnx
from functools import partial
import optax
from datasets import load_dataset
import jax.numpy as jnp

from tqdm.auto import tqdm

from hax.util.data import NumpyLoader
from hax.opt import riemannian_adam

print("Loading dataset...")

# Load and preprocess the dataset with batching and channel dimension
dataset = load_dataset("ylecun/mnist").with_format("numpy")

# Batch the datasets
batch_size = 32

train_loader = NumpyLoader(dataset["train"], batch_size=batch_size)
eval_loader = NumpyLoader(dataset["test"], batch_size=batch_size)

eval_ds = dataset["test"].iter(batch_size=batch_size)


class CNN(nnx.Module):
    """A simple CNN model with corrected dimensions."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(
            64 * 7 * 7, 256, rngs=rngs
        )  # Corrected input features
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = self.avg_pool(x)
        x = nnx.relu(self.conv2(x))
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten to (batch_size, 64*7*7)
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


print("Creating model...")
# Instantiate the model with corrected architecture
model = CNN(rngs=nnx.Rngs(0))
learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, riemannian_adam(learning_rate))
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)


def loss_fn(model: CNN, batch):
    logits = model(jnp.expand_dims(batch["image"], -1))
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])
    optimizer.update(grads)


@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}

eval_every = 5  # Evaluate every 100 steps for efficiency


print("Starting training..")


def train_single_epoch():
    for batch in tqdm(
        train_loader,
        desc="Train",
        leave=False,
        total=len(train_loader),
    ):
        # Convert images to correct shape if necessary (handled in preprocessing)
        train_step(model, optimizer, metrics, batch)


def eval_single_epoch():
    for batch in tqdm(
        eval_loader,
        desc="Eval",
        leave=False,
        total=len(eval_loader),
    ):
        # Convert images to correct shape if necessary (handled in preprocessing)
        eval_step(model, metrics, batch)


num_epochs = 100
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    train_single_epoch()

    msg = f"[{epoch + 1}/{num_epochs}]"

    # Training metrics
    train_metrics = metrics.compute()
    for metric, value in train_metrics.items():
        msg = f"{msg} train_{metric}: {value:.4f}"
        metrics_history[f"train_{metric}"].append(value)
    metrics.reset()

    eval_single_epoch()
    # Eval metrics
    train_metrics = metrics.compute()
    for metric, value in train_metrics.items():
        msg = f"{msg} test_{metric}: {value:.4f}"
        metrics_history[f"test_{metric}"].append(value)
    metrics.reset()

    tqdm.write(msg)
