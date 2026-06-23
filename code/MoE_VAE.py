from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import os
import numpy as np


class VAE(LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim,
        encoder_hidden_dims,
        decoder_hidden_dims,
        activation="leakyrelu",
        use_norm=False,
        use_dropout=False,
        use_softplus_output=False,
        **kwargs,
    ):
        super().__init__()
        # Define the activation function
        self.use_softplus_output = use_softplus_output
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Encoder layers
        self.encoder_layers = self.build_layers(
            input_dim, encoder_hidden_dims, use_norm, use_dropout
        )
        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        # Decoder layers
        self.decoder_layers = self.build_layers(
            latent_dim, decoder_hidden_dims, use_norm, use_dropout
        )
        # self.decoder_layers.add_module('softplus', nn.Softplus())
        self.decoder_layers.add_module(
            "output_layer", nn.Linear(decoder_hidden_dims[-1], output_dim)
        )
        if self.use_softplus_output:
            self.decoder_layers.add_module("output_activation", nn.Softplus())
        # self.decoder_layers.add_module('output_activation', nn.Tanh())  # Assuming output is in range [-1, 1]
        # with the classic robust preprocessing method it is -1 to 1, but for others it may not.

    def build_layers(self, input_dim, hidden_dims, use_norm, use_dropout=False):
        layers = []
        current_size = input_dim
        for hidden_dim in hidden_dims:
            next_size = hidden_dim
            layers.append(nn.Linear(current_size, next_size))
            if use_norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_norm == "layer":
                layers.append(nn.LayerNorm(hidden_dim))
            elif use_norm == "group":
                num_groups = max(1, hidden_dim // 4)
                layers.append(
                    nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim)
                )
            layers.append(self.activation)
            if use_dropout:
                layers.append(nn.Dropout(0.1))
            current_size = next_size
        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.encoder_layers(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder_layers(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        pred_y = self.decode(z)
        return {"pred_y": pred_y, "mu": mu, "log_var": log_var}

    def loss_fn(self, output_dict, kld_weight=0.0):
        pred_y, y, mu, log_var = (
            output_dict["pred_y"],
            output_dict["y"],
            output_dict["mu"],
            output_dict["log_var"],
        )
        batch_size = y.shape[0]
        MAE = F.l1_loss(pred_y, y, reduction="mean")
        # Reconstruction loss (MSE)
        MSE = F.mse_loss(pred_y, y, reduction="mean")
        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        # Return combined loss
        return {
            "total_loss": MAE + kld_weight * KLD,
            "mae_loss": MAE,
            "mse_loss": MSE,
            "kld_loss": KLD,
        }


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_dim]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts

        # Safety check: ensure at least one example per expert
        if (gates.sum(dim=0) == 0).any():
            # Find experts with no assignments and create dummy assignments
            empty_experts = (gates.sum(dim=0) == 0).nonzero().squeeze(1)
            if empty_experts.numel() > 0:
                # Assign the first example to all empty experts with a small weight
                for expert_idx in empty_experts:
                    gates[0, expert_idx] = 1e-5

        # Sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # Drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # Get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # Calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()

        # Safety check: ensure no expert has 0 examples
        for i, size in enumerate(self._part_sizes):
            if size == 0:
                # Add a dummy example to this expert
                self._part_sizes[i] = 1
                if i >= len(self._expert_index):
                    # Add a new dummy index if needed
                    self._expert_index = torch.cat(
                        [self._expert_index, torch.tensor([[i]], device=gates.device)]
                    )
                    self._batch_index = torch.cat(
                        [self._batch_index, torch.tensor([0], device=gates.device)]
                    )

        # Expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

        # Safety check for nonzero gates
        if (self._nonzero_gates <= 0).any():
            self._nonzero_gates = torch.clamp(self._nonzero_gates, min=1e-5)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=stitched.device,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE_VAE(LightningModule):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_dim: integer - size of the input
    output_dim: integer - size of the input
    num_experts: an integer - number of experts
    hidden_dims: an integer - hidden_dims size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim,
        encoder_hidden_dims,
        decoder_hidden_dims,
        num_experts,
        k=4,
        activation="leakyrelu",
        noisy_gating=True,
        use_norm=False,
        use_dropout=False,
        use_softplus_output=False,
        **kwargs,
    ):
        super(MoE_VAE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.num_experts = num_experts
        self.k = k
        self.activation = activation
        self.use_norm = use_norm
        self.use_dropout = use_dropout
        self.use_softplus_output = use_softplus_output

        # instantiate experts
        self.experts = nn.ModuleList(
            [
                VAE(
                    self.input_dim,
                    self.output_dim,
                    self.latent_dim,
                    self.encoder_hidden_dims,
                    self.decoder_hidden_dims,
                    self.activation,
                    use_norm=self.use_norm,
                    use_dropout=self.use_dropout,
                    use_softplus_output=self.use_softplus_output,
                )
                for i in range(self.num_experts)
            ]
        )

        self.w_gate = nn.Parameter(
            torch.zeros(input_dim, num_experts, dtype=self.dtype), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(input_dim, num_experts, dtype=self.dtype), requires_grad=True
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.batch_gates = None

        assert self.k <= self.num_experts

    def forward(self, x, moe_weight=1e-2):
        """Args:
        x: tensor shape [batch_size, input_dim]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_dim].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        self.batch_gates = gates
        # calculate importance loss
        importance = gates.sum(0)

        moe_loss = moe_weight * self.cv_squared(
            importance
        ) + moe_weight * self.cv_squared(load)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = []
        for i in range(self.num_experts):
            input_i = expert_inputs[i]
            if input_i.shape[0] > 1:
                expert_outputs.append(self.experts[i](input_i)["pred_y"])
            else:
                expert_outputs.append(
                    torch.zeros(
                        (input_i.shape[0], self.output_dim), device=input_i.device
                    )
                )
        pred_y = dispatcher.combine(expert_outputs)
        return {"pred_y": pred_y, "moe_loss": moe_loss}

    def loss_fn(self, output_dict) -> torch.Tensor:
        """
        Compute loss between model output and target.

        Args:
            output: Model output tensor of shape (batch, output_dim)
            target: Target tensor of shape (batch, output_dim)

        Returns:
            loss: Scalar tensor representing the loss
        """
        pred_y = output_dict["pred_y"]
        y = output_dict["y"]
        batch_size = y.shape[0]
        MAE = F.l1_loss(pred_y, y, reduction="mean")
        mse_losss = F.mse_loss(pred_y, y, reduction="mean")
        moe_loss = output_dict.get(
            "moe_loss", torch.tensor(0.0, device=pred_y.device, dtype=pred_y.dtype)
        )
        total_loss = MAE + moe_loss
        return {
            "total_loss": total_loss,
            "mae_loss": MAE,
            "mse_loss": mse_losss,
            "moe_loss": moe_loss,
        }

    def get_batch_gates(self):
        return self.batch_gates

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_dim]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits
            # Add this safety check to ensure we always have at least one expert selected
        if (logits.sum(dim=1) == 0).any():
            # Add a small positive value to ensure we have non-zero logits
            logits = logits + 1e-5

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True, dtype=self.dtype)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # Safety check - ensure at least one expert is selected per sample
        if (gates.sum(dim=1) < 1e-6).any():
            # Force selection of the top expert for samples with no experts
            problematic_samples = (gates.sum(dim=1) < 1e-6).nonzero().squeeze(1)
            if problematic_samples.numel() > 0:  # If there are problematic samples
                # Select the top expert for these samples
                top_expert = top_indices[problematic_samples, 0]
                # Set a minimum value for the gate
                gates[problematic_samples, top_expert] = 0.1

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


class MoE_VAE_Token(LightningModule):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_dim: integer - size of the input
    output_dim: integer - size of the input
    num_experts: an integer - number of experts
    hidden_dims: an integer - hidden_dims size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim,
        encoder_hidden_dims,
        decoder_hidden_dims,
        num_experts,
        k=4,
        activation="leakyrelu",
        noisy_gating=True,
        use_norm=False,
        use_dropout=False,
        use_softplus_output=False,
        **kwargs,
    ):
        super(MoE_VAE_Token, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.activation = activation
        self.use_norm = use_norm
        self.use_dropout = use_dropout
        self.use_softplus_output = use_softplus_output

        # instantiate experts
        self.sub_input_dims = [input_dim // num_experts] * (num_experts - 1)
        self.sub_input_dims.append(input_dim - sum(self.sub_input_dims))

        self.experts = nn.ModuleList(
            [
                VAE(
                    sub_dim,
                    sub_dim,
                    self.latent_dim,
                    self.encoder_hidden_dims,
                    self.decoder_hidden_dims,
                    self.activation,
                    use_norm=self.use_norm,
                    use_dropout=self.use_dropout,
                    use_softplus_output=self.use_softplus_output,
                )
                for sub_dim in self.sub_input_dims
            ]
        )

        self.w_gate = nn.Parameter(
            torch.zeros(input_dim, num_experts, dtype=self.dtype), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(input_dim, num_experts, dtype=self.dtype), requires_grad=True
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.batch_gates = None
        self.k = k
        assert self.k <= self.num_experts

    def forward(self, x, moe_weight=0.0):
        """
        Token-wise MoE forward pass:
        Each expert processes a different spectral segment of the input.

        Args:
            x: Tensor of shape [batch_size, input_dim]
            moe_weight: kept for compatibility, but unused in token-wise mode.

        Returns:
            A dict with:
                'pred_y': reconstructed tensor of shape [batch_size, input_dim]
                'moe_loss': dummy 0.0 (no gating loss in token-wise)
        """
        # Split the input into band segments for each expert
        x_chunks = torch.split(x, self.sub_input_dims, dim=1)

        expert_outputs = []
        for i in range(self.num_experts):
            out_i = self.experts[i](x_chunks[i])["pred_y"]
            expert_outputs.append(out_i)

        pred_y = torch.cat(expert_outputs, dim=1)
        return {
            "pred_y": pred_y,
            "moe_loss": torch.tensor(0.0, device=x.device, dtype=x.dtype),
        }

    def loss_fn(self, output_dict) -> torch.Tensor:
        """
        Compute loss between model output and target.

        Args:
            output: Model output tensor of shape (batch, output_dim)
            target: Target tensor of shape (batch, output_dim)

        Returns:
            loss: Scalar tensor representing the loss
        """
        pred_y = output_dict["pred_y"]
        y = output_dict["y"]
        batch_size = y.shape[0]
        MAE = F.l1_loss(pred_y, y, reduction="mean")
        mse_losss = F.mse_loss(pred_y, y, reduction="mean")
        moe_loss = output_dict.get(
            "moe_loss", torch.tensor(0.0, device=pred_y.device, dtype=pred_y.dtype)
        )
        total_loss = MAE + moe_loss
        return {
            "total_loss": total_loss,
            "mae_loss": MAE,
            "mse_loss": mse_losss,
            "moe_loss": moe_loss,
        }


def train(model, train_dl, device, epochs=200, optimizer=None, save_dir=None):
    model.train()
    min_total_loss = float("inf")
    best_model_path = os.path.join(save_dir, "best_model_minloss.pth")

    total_list = []
    l1_list = []

    for epoch in range(epochs):
        total_loss_epoch = 0.0
        l1_epoch = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            output_dict = model(x)
            output_dict["y"] = y

            loss_dict = model.loss_fn(output_dict)

            loss = loss_dict["total_loss"]
            l1 = loss_dict["mae_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            l1_epoch += l1.item()

        avg_total_loss = total_loss_epoch / len(train_dl)
        avg_l1 = l1_epoch / len(train_dl)

        print(f"[Epoch {epoch+1}] Total: {avg_total_loss:.4f} | L1: {avg_l1:.4f}")
        total_list.append(avg_total_loss)
        l1_list.append(avg_l1)

        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            torch.save(model.state_dict(), best_model_path)

    return {"total_loss": total_list, "l1_loss": l1_list, "best_loss": min_total_loss}


def evaluate(model, test_dl, device, TSS_scalers_dict=None, log_offset=0.01):
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            output_dict = model(x)
            y_pred = output_dict["pred_y"]
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # === Inverse transformation ===
    if TSS_scalers_dict is not None:
        log_scaler = TSS_scalers_dict["log"]
        robust_scaler = TSS_scalers_dict["robust"]

        # First reverse min-max, then reverse log
        predictions_inverse = (
            log_scaler.inverse_transform(
                torch.tensor(
                    robust_scaler.inverse_transform(
                        torch.tensor(predictions, dtype=torch.float32)
                    ),
                    dtype=torch.float32,
                )
            )
            .numpy()
            .flatten()
        )

        actuals_inverse = (
            log_scaler.inverse_transform(
                torch.tensor(
                    robust_scaler.inverse_transform(
                        torch.tensor(actuals, dtype=torch.float32)
                    ),
                    dtype=torch.float32,
                )
            )
            .numpy()
            .flatten()
        )
    else:
        predictions_inverse = (10 ** predictions.flatten()) - log_offset
        actuals_inverse = (10 ** actuals.flatten()) - log_offset

    return predictions_inverse, actuals_inverse


def evaluate_token(model, test_dl, device, TSS_scalers_dict=None, log_offset=0.01):
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            output_dict = model(x)
            y_pred = output_dict["pred_y"]  # [B, token_len]

            if y_pred.ndim == 2:
                y_pred = y_pred.mean(dim=1, keepdim=True)  # [B, 1]

            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # === Inverse transformation ===
    if TSS_scalers_dict is not None:
        log_scaler = TSS_scalers_dict["log"]
        robust_scaler = TSS_scalers_dict["robust"]

        # First reverse min-max, then reverse log
        predictions_inverse = (
            log_scaler.inverse_transform(
                torch.tensor(
                    robust_scaler.inverse_transform(
                        torch.tensor(predictions, dtype=torch.float32)
                    ),
                    dtype=torch.float32,
                )
            )
            .numpy()
            .flatten()
        )

        actuals_inverse = (
            log_scaler.inverse_transform(
                torch.tensor(
                    robust_scaler.inverse_transform(
                        torch.tensor(actuals, dtype=torch.float32)
                    ),
                    dtype=torch.float32,
                )
            )
            .numpy()
            .flatten()
        )
    else:
        predictions_inverse = (10 ** predictions.flatten()) - log_offset
        actuals_inverse = (10 ** actuals.flatten()) - log_offset

    return predictions_inverse, actuals_inverse
