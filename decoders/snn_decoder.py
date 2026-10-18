"""
Spiking Neural Network Decoder.

Single-layer fully connected SNN with Leaky Integrate-and-Fire (LIF) neurons
and synaptic conductance, trained with surrogate gradient descent.

Reference: Donati/Valle et al. (2026), Nature Communications
    Architecture: 56 input -> N_class output LIF neurons (Synaptic model)
    Training: Surrogate gradient (Heaviside), MSE spike count loss, Adam
    Key params: alpha=0.95 (syn decay), beta=0.98 (mem decay), Uthr=0.5

Adapted for our use case:
    Input: spike trains from organoid recordings (n_channels, n_timesteps)
    Output: classification of which stimulus image was presented
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class SynapticLIF(nn.Module):
    """
    Second-order LIF neuron layer with synaptic conductance.

    Implements the Synaptic model from snnTorch:
        I_syn[t] = alpha * I_syn[t-1] + I[t]
        U[t] = beta * U[t-1] + I_syn[t] - S_out[t-1] * U_thr
        S_out[t] = Heaviside(U[t] - U_thr)

    Uses surrogate gradient (sigmoid) for backprop through Heaviside.
    """

    def __init__(self, alpha=0.95, beta=0.98, threshold=0.5):
        """
        Args:
            alpha: Synaptic current decay rate = exp(-dt/tau_syn).
            beta: Membrane potential decay rate = exp(-dt/tau_mem).
            threshold: Spike threshold.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def forward(self, input_current, syn, mem):
        """
        Single time step update.

        Args:
            input_current: (batch, n_neurons) weighted input.
            syn: (batch, n_neurons) previous synaptic current.
            mem: (batch, n_neurons) previous membrane potential.

        Returns:
            spk: (batch, n_neurons) binary spike output.
            syn: Updated synaptic current.
            mem: Updated membrane potential.
        """
        syn = self.alpha * syn + input_current
        mem = self.beta * mem + syn - self._spike_fn(mem) * self.threshold
        spk = self._spike_fn(mem)
        return spk, syn, mem

    def _spike_fn(self, mem):
        """Heaviside with surrogate gradient (sigmoid)."""
        return SurrogateSpike.apply(mem - self.threshold)

    def init_state(self, batch_size, n_neurons, device="cpu"):
        """Initialize synaptic current and membrane potential to zero."""
        syn = torch.zeros(batch_size, n_neurons, device=device)
        mem = torch.zeros(batch_size, n_neurons, device=device)
        return syn, mem


class SurrogateSpike(torch.autograd.Function):
    """Heaviside step function with sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Sigmoid surrogate gradient with slope factor
        slope = 25.0
        sigmoid = torch.sigmoid(slope * x)
        return grad_output * sigmoid * (1 - sigmoid) * slope


class SNNDecoder(nn.Module):
    """
    Single-layer SNN decoder for spike train classification.

    Architecture:
        Input spike trains (n_channels, n_timesteps)
        -> Linear projection (n_channels -> n_classes)
        -> Synaptic LIF neurons (n_classes)
        -> Spike count per output neuron -> argmax = predicted class

    Training:
        Loss: MSE between output spike counts and target counts
        Target: correct neuron fires at target_rate, others at incorrect_rate
    """

    def __init__(
        self,
        n_input,
        n_classes,
        alpha=0.95,
        beta=0.98,
        threshold=0.5,
    ):
        """
        Args:
            n_input: Number of input channels.
            n_classes: Number of output classes.
            alpha: Synaptic current decay (Donati: 0.95).
            beta: Membrane potential decay (Donati: 0.98).
            threshold: Spike threshold (Donati: 0.5).
        """
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes

        self.fc = nn.Linear(n_input, n_classes)
        self.lif = SynapticLIF(alpha=alpha, beta=beta, threshold=threshold)

    def forward(self, spike_input):
        """
        Forward pass over time steps.

        Args:
            spike_input: (batch, n_timesteps, n_input) spike trains.

        Returns:
            spike_counts: (batch, n_classes) total spike counts per output neuron.
            membrane_trace: (batch, n_timesteps, n_classes) membrane potentials.
        """
        batch_size, n_timesteps, n_input = spike_input.shape
        device = spike_input.device

        syn, mem = self.lif.init_state(batch_size, self.n_classes, device)
        spike_counts = torch.zeros(batch_size, self.n_classes, device=device)
        membrane_trace = []

        for t in range(n_timesteps):
            x = self.fc(spike_input[:, t, :])
            spk, syn, mem = self.lif(x, syn, mem)
            spike_counts += spk
            membrane_trace.append(mem.clone())

        membrane_trace = torch.stack(membrane_trace, dim=1)
        return spike_counts, membrane_trace

    def predict(self, spike_input):
        """Return predicted class labels."""
        spike_counts, _ = self.forward(spike_input)
        return spike_counts.argmax(dim=1)


def train_snn_decoder(
    model,
    train_loader,
    val_loader,
    n_epochs=500,
    lr=1e-3,
    target_rate=0.8,
    incorrect_rate=0.001,
    device="cpu",
):
    """
    Train SNN decoder with MSE spike count loss.

    Donati et al.: target neuron fires at 80% probability per timestep,
    non-target neurons at 0.1%.

    Args:
        model: SNNDecoder instance.
        train_loader: Yields (spike_trains, labels).
            spike_trains: (batch, n_timesteps, n_channels).
            labels: (batch,) integer class labels.
        val_loader: Validation data.
        n_epochs: Training epochs (Donati: 500).
        lr: Learning rate (Donati: 1e-3).
        target_rate: Target firing rate for correct class neuron.
        incorrect_rate: Target rate for incorrect class neurons.
        device: Torch device.

    Returns:
        model: Trained model.
        history: Dict with loss and accuracy per epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(n_epochs):
        # Training
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for spike_trains, labels in train_loader:
            spike_trains = spike_trains.to(device)
            labels = labels.to(device)

            n_timesteps = spike_trains.shape[1]

            # Build target spike count
            target_counts = torch.full(
                (labels.shape[0], model.n_classes),
                incorrect_rate * n_timesteps,
                device=device,
            )
            for i, lbl in enumerate(labels):
                target_counts[i, lbl] = target_rate * n_timesteps

            optimizer.zero_grad()
            spike_counts, _ = model(spike_trains)
            loss = torch.mean((spike_counts - target_counts) ** 2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            predicted = spike_counts.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.shape[0]

        history["train_loss"].append(total_loss / total)
        history["train_acc"].append(correct / total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for spike_trains, labels in val_loader:
                spike_trains = spike_trains.to(device)
                labels = labels.to(device)
                n_timesteps = spike_trains.shape[1]

                target_counts = torch.full(
                    (labels.shape[0], model.n_classes),
                    incorrect_rate * n_timesteps,
                    device=device,
                )
                for i, lbl in enumerate(labels):
                    target_counts[i, lbl] = target_rate * n_timesteps

                spike_counts, _ = model(spike_trains)
                loss = torch.mean((spike_counts - target_counts) ** 2)
                val_loss += loss.item() * labels.shape[0]
                predicted = spike_counts.argmax(dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.shape[0]

        history["val_loss"].append(val_loss / val_total)
        history["val_acc"].append(val_correct / val_total)

    return model, history
