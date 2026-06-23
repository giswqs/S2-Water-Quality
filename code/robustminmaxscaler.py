import torch


class RobustMinMaxScaler:
    """
    Robust MinMax scaler for PyTorch tensors with quantile-based scaling option.

    Args:
        feature_range: Desired range (min, max) for scaling
        global_scale: If True, scale using global quantiles across all features.
                     If False, scale each feature independently.
        robust: If True, use quantiles instead of min/max for robust scaling
        quantile_range: Tuple of (lower, upper) quantiles to use (e.g., (0.05, 0.95))
        clip_outliers: If True, clip values outside quantile range to the quantile values
    """

    def __init__(
        self,
        feature_range=(0, 1),
        global_scale=True,
        robust=True,
        quantile_range=(0.05, 0.95),
        clip_outliers=False,
    ):
        self.feature_range = feature_range
        self.global_scale = global_scale
        self.robust = robust
        self.quantile_range = quantile_range
        self.clip_outliers = clip_outliers
        self.min_val = None
        self.max_val = None
        self.fitted = False

    def fit(self, X):
        """
        Fit the scaler to the data.

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, features]
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        if self.robust:
            # Use quantiles for robust scaling
            if self.global_scale:
                # Global quantiles across all values
                flat_X = X.flatten()
                # If tensor is too large, use sampling for quantile calculation
                max_samples = 1000000  # 1M samples max
                if len(flat_X) > max_samples:
                    # Randomly sample from the tensor
                    indices = torch.randperm(len(flat_X))[:max_samples]
                    sampled_X = flat_X[indices]
                    self.min_val = torch.quantile(sampled_X, self.quantile_range[0])
                    self.max_val = torch.quantile(sampled_X, self.quantile_range[1])
                else:
                    self.min_val = torch.quantile(flat_X, self.quantile_range[0])
                    self.max_val = torch.quantile(flat_X, self.quantile_range[1])
            else:
                # Feature-wise quantiles
                self.min_val = torch.quantile(
                    X, self.quantile_range[0], dim=0, keepdim=True
                )
                self.max_val = torch.quantile(
                    X, self.quantile_range[1], dim=0, keepdim=True
                )
        else:
            # Use traditional min/max
            if self.global_scale:
                # Global min/max across all values
                self.min_val = torch.min(X)
                self.max_val = torch.max(X)
            else:
                # Feature-wise min/max
                self.min_val = torch.min(X, dim=0, keepdim=True)[0]
                self.max_val = torch.max(X, dim=0, keepdim=True)[0]

        self.fitted = True
        return self

    def transform(self, X):
        """
        Transform the data using fitted parameters.

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, features]

        Returns:
            torch.Tensor: Scaled tensor
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        # Clip outliers if using robust scaling
        if self.robust and self.clip_outliers:
            X = torch.clamp(X, min=self.min_val, max=self.max_val)

        # Avoid division by zero
        range_val = self.max_val - self.min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)

        # Scale to [0, 1]
        scaled = (X - self.min_val) / range_val

        # Scale to desired range
        min_target, max_target = self.feature_range
        return scaled * (max_target - min_target) + min_target

    def fit_transform(self, X):
        """
        Fit the scaler and transform the data in one step.

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, features]

        Returns:
            torch.Tensor: Scaled tensor
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Inverse transform the scaled data back to original scale.

        Args:
            X (torch.Tensor): Scaled tensor

        Returns:
            torch.Tensor: Original scale tensor
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        min_target, max_target = self.feature_range

        # Scale back to [0, 1]
        normalized = (X - min_target) / (max_target - min_target)

        # Scale back to original range
        range_val = self.max_val - self.min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)

        return normalized * range_val + self.min_val
