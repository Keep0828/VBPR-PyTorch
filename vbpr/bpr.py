"""This module contains a BPR implementation in PyTorch."""

from typing import Optional, Tuple, Union, cast

import torch
from torch import nn


class BPR(nn.Module):
    """This class represents the BPR model as a PyTorch nn.Module.

    The implementation follows the specifications of the original paper:
    'BPR: Visual Bayesian Personalized Ranking from Implicit Feedback'

    NOTE: The model contains the (pretrained)  visual features as a layer to
    improve performance. Another possible implementation of this would be to
    store the features in the Dataset class and pass the emebddings to the
    forward method."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim_gamma: int
    ):
        super().__init__()

        # Latent factors (gamma)
        self.gamma_users = nn.Embedding(n_users, dim_gamma)
        self.gamma_items = nn.Embedding(n_items, dim_gamma)

        # Random weight initialization
        self.reset_parameters()

    def forward(
        self, ui: torch.Tensor, pi: torch.Tensor, ni: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Feed forward a given input (batch). Each object is expected
        to be a Tensor.

        Args:
            ui: User index, as a Tensor.
            pi: Positive item index, as a Tensor.
            ni: Negative item index, as a Tensor.

        Returns:
            Network output (scalar) for each input.
        """
        # User
        ui_latent_factors = self.gamma_users(ui)  # Latent factors of user u

        # Items
        pi_latent_factors = self.gamma_items(pi)  # Pos. item visual factors
        ni_latent_factors = self.gamma_items(ni)  # Neg. item visual factors

        # Precompute differences
        diff_latent_factors = pi_latent_factors - ni_latent_factors
        # x_uij
        x_uij = (
                (ui_latent_factors * diff_latent_factors).sum(dim=1).unsqueeze(-1)
        )

        return cast(torch.Tensor, x_uij.squeeze())

    @torch.no_grad()
    def recommend(
        self,
        users: torch.Tensor,
        items: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Predict score for interactions.

        Calculates the score for the given user-item interactions.
        Output shape matches the shape of `users * items`

        Args:
            users: Users indices, as a Tensor.
            items: Items indices, as a Tensor.
            cache: Optional. A precalculated tuple of Tensors.

        Returns:
            Prediction score for each user-item pair.
        """

        def check_input_tensor(tensor: torch.Tensor, name: str) -> None:
            if tensor.dim() != 2:
                raise ValueError(f"{name} tensor must have exactly two dimensions")
            elif not any(size == 1 for size in tensor.size()):
                raise ValueError(f"{name} tensor must contain a singleton dimension")

        check_input_tensor(users, "users")
        if items is not None:
            check_input_tensor(items, "items")

        use_matmul = False
        if items is None or (
            users.size(0) == items.size(1) == 1 or users.size(1) == items.size(0) == 1
        ):
            use_matmul = True
        elif users.size() != items.size():
            raise ValueError(
                "users and items must have equal shape or different singleton dimension"
            )

        items_selector: Union[slice, torch.Tensor] = (
            slice(None) if items is None else items
        )
        repeat_factors = (max(users.size()), 1) if use_matmul else (1, 1)

        u_latent_factors = self.gamma_users(users).squeeze()

        if u_latent_factors.dim() == 1:
            u_latent_factors = u_latent_factors.unsqueeze(0)

        i_latent_factors = self.gamma_items.weight[items_selector]
        i_latent_factors = i_latent_factors.squeeze()
        if i_latent_factors.dim() == 1:
            i_latent_factors = i_latent_factors.unsqueeze(0)

        if use_matmul:
            latent_component = torch.matmul(u_latent_factors, i_latent_factors.T)
        else:
            latent_component = (
                (u_latent_factors * i_latent_factors).sum(dim=1).unsqueeze(0)
            )

        x_ui = latent_component
        if items is None:
            if x_ui.size() != (users.size(0), self.gamma_items.weight.size(0)):
                x_ui = x_ui.T
        elif x_ui.size() == (users.size(1), items.size(0)):
            x_ui = x_ui.T
        return cast(torch.Tensor, x_ui)

    def reset_parameters(self) -> None:
        """Resets network weights.

        Restart network weights using a Xavier uniform distribution.
        """
        # Latent factors (gamma)
        nn.init.xavier_uniform_(self.gamma_users.weight)
        nn.init.xavier_uniform_(self.gamma_items.weight)

    def generate_cache(
        self, grad_enabled: bool = False
    ):
        return None
