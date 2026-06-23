import torch


class RobustMinMaxMapper:
    """
    Robust MinMax mapper for PyTorch tensors that maps data X to match reference data's quantile range.

    Args:
        global_scale: If True, use global quantiles across all features.
                     If False, map each feature independently.
        robust: If True, use quantiles instead of min/max for robust mapping
        quantile_range: Tuple of (lower, upper) quantiles to use (e.g., (0.05, 0.95))
        clip_outliers: If True, clip values outside quantile range to the quantile values
    """

    def __init__(
        self,
        global_scale=True,
        robust=True,
        quantile_range=(0.05, 0.95),
        clip_outliers=False,
    ):
        self.global_scale = global_scale
        self.robust = robust
        self.quantile_range = quantile_range
        self.clip_outliers = clip_outliers
        self.x_min = None
        self.x_max = None
        self.ref_min = None
        self.ref_max = None
        self.fitted = False

    def _compute_range(self, X):
        """Helper method to compute min/max or quantiles for a tensor."""
        if self.robust:
            if self.global_scale:
                # Global quantiles across all values
                flat_X = X.flatten()
                # If tensor is too large, use sampling for quantile calculation
                max_samples = 1000000  # 1M samples max
                if len(flat_X) > max_samples:
                    # Randomly sample from the tensor
                    indices = torch.randperm(len(flat_X))[:max_samples]
                    sampled_X = flat_X[indices]
                    min_val = torch.quantile(sampled_X, self.quantile_range[0])
                    max_val = torch.quantile(sampled_X, self.quantile_range[1])
                else:
                    min_val = torch.quantile(flat_X, self.quantile_range[0])
                    max_val = torch.quantile(flat_X, self.quantile_range[1])
            else:
                # Feature-wise quantiles
                min_val = torch.quantile(X, self.quantile_range[0], dim=0, keepdim=True)
                max_val = torch.quantile(X, self.quantile_range[1], dim=0, keepdim=True)
        else:
            # Use traditional min/max
            if self.global_scale:
                # Global min/max across all values
                min_val = torch.min(X)
                max_val = torch.max(X)
            else:
                # Feature-wise min/max
                min_val = torch.min(X, dim=0, keepdim=True)[0]
                max_val = torch.max(X, dim=0, keepdim=True)[0]

        return min_val, max_val

    def fit(self, X, reference_data):
        """
        Fit the mapper to map X's range to reference_data's range.

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, features]
            reference_data (torch.Tensor): Reference tensor of shape [batch_size, features]
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(reference_data, torch.Tensor):
            reference_data = torch.tensor(reference_data, dtype=torch.float32)

        # Compute ranges for both X and reference data
        self.x_min, self.x_max = self._compute_range(X)
        self.ref_min, self.ref_max = self._compute_range(reference_data)

        self.fitted = True
        return self

    def transform(self, X):
        """
        Transform the data X to match reference data's range.

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, features]

        Returns:
            torch.Tensor: Mapped tensor with reference data's range
        """
        if not self.fitted:
            raise ValueError("Mapper must be fitted before transform")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        # Clip outliers if using robust scaling and clip_outliers is True
        if self.robust and self.clip_outliers:
            X = torch.clamp(X, min=self.x_min, max=self.x_max)

        # Avoid division by zero
        x_range = self.x_max - self.x_min
        x_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)

        # Normalize X to [0, 1] based on its original range
        normalized = (X - self.x_min) / x_range

        # Map to reference data's range
        ref_range = self.ref_max - self.ref_min
        transformed = normalized * ref_range + self.ref_min

        # Additional clipping to ensure values are within reference range
        if self.clip_outliers:
            transformed = torch.clamp(transformed, min=self.ref_min, max=self.ref_max)

        return transformed

    def fit_transform(self, X, reference_data):
        """
        Fit the mapper and transform the data in one step.

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, features]
            reference_data (torch.Tensor): Reference tensor of shape [batch_size, features]

        Returns:
            torch.Tensor: Mapped tensor
        """
        return self.fit(X, reference_data).transform(X)

    def inverse_transform(self, X):
        """
        Inverse transform the mapped data back to original X scale.

        Args:
            X (torch.Tensor): Mapped tensor

        Returns:
            torch.Tensor: Original scale tensor
        """
        if not self.fitted:
            raise ValueError("Mapper must be fitted before inverse_transform")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        # Normalize back from reference range to [0, 1]
        ref_range = self.ref_max - self.ref_min
        ref_range = torch.where(ref_range == 0, torch.ones_like(ref_range), ref_range)
        normalized = (X - self.ref_min) / ref_range

        # Scale back to original X range
        x_range = self.x_max - self.x_min
        return normalized * x_range + self.x_min

    def get_params(self):
        """
        Get the fitted parameters.

        Returns:
            dict: Dictionary containing fitted parameters
        """
        if not self.fitted:
            raise ValueError("Mapper must be fitted before getting parameters")

        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "ref_min": self.ref_min,
            "ref_max": self.ref_max,
            "quantile_range": self.quantile_range if self.robust else None,
            "robust": self.robust,
            "global_scale": self.global_scale,
        }
