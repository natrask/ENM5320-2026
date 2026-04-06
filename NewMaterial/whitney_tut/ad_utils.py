import torch
import torch_geometric.nn as gnn
import scipy
import scipy.special as sp
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np


@torch.no_grad()
def box_init(layer):
    p = torch.rand(layer.weight.shape)
    n_hat = torch.randn(layer.weight.shape)
    n = n_hat / torch.norm(n_hat, 2, dim=1, keepdim=True)
    p_max = torch.nn.functional.relu(n / n.abs())
    k = 1.0 / ((p_max - p) * n).sum(dim=1, keepdim=True)
    A = k * n
    b = (k * (n * p).sum(dim=1, keepdim=True)).squeeze()
    layer.weight = nn.Parameter(A)
    layer.bias = nn.Parameter(b)


@torch.no_grad()
def box_init_resnet(layer, l, L):
    if l == 1:
        box_init(layer)
    else:
        m = (l + 1 / (L - 1)) ** l
        p = m * torch.rand(layer.weight.shape)
        n_hat = torch.randn(layer.weight.shape)
        n = n_hat / torch.norm(n_hat, 2, dim=1, keepdim=True)
        p_max = m * torch.nn.functional.relu(n / n.abs())
        k = 1.0 / ((p_max - p) * n * (L - 1)).sum(dim=1, keepdim=True)
        A = k * n
        b = (k * (n * p).sum(dim=1, keepdim=True)).squeeze()
        layer.weight = nn.Parameter(A)
        layer.bias = nn.Parameter(b)


def construct_W(W_trainable):
    """
    Construct the matrix W which maps the nodal basis functions of the fine mesh to the basis functions
    that comprise the "coarse mesh". W is a matrix of size Npou x M, where Npou is the number of nodal basis
    functions on the coarse mesh and M is the number of basis functions on the fine mesh. The entries of each
    column of W must sum to 1 so that the coarse mesh remains a partition of unity (POU) of space.

    That is, the i^th Whitney 0-form on the coarse mesh is given by psi^0_i(x) = \sum_j W_ij @ phi^0_j(x),
    with \sum_i psi^0_i(x) = 1 for all x.

    This function takes in only the middle trainable block of the W matrix; the top and bottom rows are set to
    [1, 0, 0, ..., 0] and [0, ..., 0, 0, 1], respectively. This sets the first and last psi_0 functions to be
    identical to the phi_0 functions on the corresponding boundaries.

    Args:
        W_trainable (torch.Tensor): The trainable matrix W of size (Npou-2) x (M-2).

    Returns:
        torch.Tensor: The matrix W of size Npou x M
    """
    Npou = W_trainable.shape[0] + 2
    M = W_trainable.shape[1] + 2
    W_softmax = torch.softmax(W_trainable, dim=0)
    W_toprow = torch.zeros(1, M, device=W_trainable.device)
    W_toprow[0, 0] = 1.0
    W_bottomrow = torch.zeros(1, M, device=W_trainable.device)
    W_bottomrow[0, -1] = 1.0
    W_middleblock = torch.cat(
        [
            torch.zeros(Npou - 2, 1, device=W_trainable.device),
            W_softmax,
            torch.zeros(Npou - 2, 1, device=W_trainable.device),
        ],
        dim=1,
    )
    W = torch.cat([W_toprow, W_middleblock, W_bottomrow], dim=0)
    return W


def phi_0(x, x_c, h, L):
    """
    Evaluate phi^0(x), centered at x_c of total width h, on the domain [0, L].
    This is just the "hat function."

    Args:
        x (torch.Tensor): The point at which to evaluate the hat function.
        x_c (float): The center of the hat function.
        h (float): (Half) the width of the hat function.
        L (float): The length of the domain.

    Returns:
        torch.Tensor: The value of this particular phi_0 function at x.
    """
    phi = torch.maximum(1.0 - torch.abs(x - x_c) / h, torch.zeros_like(x))
    phi = torch.where(torch.logical_or(x < 0, x > L), 0, phi)
    return phi


def grad_phi_0(x, x_c, h, L):
    """
    Calculate the gradient of (this particular) phi^0(x) function at a given point.

    Args:
        x (torch.Tensor): The input point.
        x_c (torch.Tensor): The center point.
        h (float): (Half) the width of the hat function.
        L (float): The length of the domain.

    Returns:
        torch.Tensor: The value of the gradient of this particular phi_0 function at x.
    """
    grad_phi = -torch.sign(x - x_c) / h
    grad_phi = torch.where(torch.abs(x - x_c) < h, grad_phi, 0)
    grad_phi = torch.where(torch.logical_or(x < 0, x > L), 0, grad_phi)
    return grad_phi


def evaluate_phi_0s(x_nodes, x):
    """
    Evaluate the set of phi_0 functions centered at the coordinates in x_nodes,
    for the given set of points x.

    Args:
        x_nodes (torch.Tensor): Set of x coordinate nodes, at each of which there is a phi_0 function centered.
        x (torch.Tensor): Set of points where each of the phi_0 functions are to be evaluated.

    Returns:
        torch.Tensor: len(x_nodes) x len(x) sized tensor containing the evaluated phi_0 functions.
    """
    h = x_nodes[1] - x_nodes[0]
    L = x_nodes[-1]
    phi_0s = torch.stack([phi_0(x, x_c, h, L) for x_c in x_nodes])
    return phi_0s


def evaluate_psi_0s(x_nodes, x, W):
    """
    Evaluate the set of psi_0 functions centered at the coordinates in x_nodes,
    for the given set of points x.

    Args:
        x_nodes (torch.Tensor): Set of x coordinate nodes, at each of which there is a phi_0 function centered.
        x (torch.Tensor): Set of points where each of the phi_0 and psi_0 functions are to be evaluated.
        W (torch.Tensor): The matrix which transforms the phi_0 functions to the psi_0 functions.

    Returns:
        torch.Tensor: Npou x len(x) sized tensor containing the evaluated phi_0 functions.

    """
    phi_0s = evaluate_phi_0s(x_nodes, x)
    psi_0s = W @ phi_0s
    return psi_0s


def evaluate_one_forms(x_nodes, x, W=None):
    """
    Evaluate the one-forms: either phi^1(x) if no W matrix is provided, or psi^1(x) if a W matrix is provided.

    Args:
        x_nodes (torch.Tensor): The nodes used for phi^0(x) basis function evaluation.
        x (torch.Tensor): The evaluation points.
        W (torch.Tensor, optional): The matrix for transforming from phi to psi bases. Defaults to None.

    Returns:
        torch.Tensor: The antisymmetric product of the basis functions and their gradients; i.e., the one-forms, in the original or transformed basis.
    """
    h = x_nodes[1] - x_nodes[0]
    L = x_nodes[-1]
    zero_forms = torch.stack([phi_0(x, x_c, h, L) for x_c in x_nodes])
    grad_zero_forms = torch.stack([grad_phi_0(x, x_c, h, L) for x_c in x_nodes])
    if W is not None:
        # zero_forms = torch.einsum("ij,jk->ik", W, zero_forms)
        zero_forms = W @ zero_forms
        # grad_zero_forms = torch.einsum("ij,jk->ik", W, grad_zero_forms)
        grad_zero_forms = W @ grad_zero_forms
    # zero_forms and grad_zero_forms are shaped like [M, num_eval_points],
    # where M is the number of basis functions in the space
    zero_forms = zero_forms.unsqueeze(1)  # shape: [M, 1, num_eval_points]
    grad_zero_forms = grad_zero_forms.unsqueeze(0)  # shape: [1, M, num_eval_points]
    # abuse broadcasting rules to get the grid of products f_i(x) * df_j(x)
    outer_product = zero_forms * grad_zero_forms  # shape: [M, M, num_eval_points]
    # however, what we want is the antisymmetric product f_i(x) * df_j(x) - f_j(x) * df_i(x)
    # this is achieved by subtracting the transpose of this tensor from itself
    outer_product = outer_product - outer_product.permute(1, 0, 2)
    # finally, we only want the linearly independent part of the basis, so we remove the diagonal (which is identically zero),
    # and everything below it (which is just -1 * its counterpart above the diagonal)
    num_zero_forms = zero_forms.shape[0]
    indices_to_pull_out = torch.ones(
        num_zero_forms, num_zero_forms, dtype=torch.bool
    ).triu(diagonal=1)
    # this indexing trick selects out all the [i, j] pairs where indices_to_pull_out[i, j] is True,
    # and arranges them into an array flattened along these first two dimensions
    outer_product = outer_product[indices_to_pull_out]
    return outer_product


def construct_quadrature(x_nodes, num_quad_pts):
    midpoints = (x_nodes[1:] + x_nodes[:-1]) / 2
    cell_widths = x_nodes[1:] - x_nodes[:-1]
    quad_pts, weights = sp.roots_legendre(num_quad_pts)
    quad_pts = torch.tensor(quad_pts, dtype=x_nodes.dtype, device=x_nodes.device)
    weights = torch.tensor(weights, dtype=x_nodes.dtype, device=x_nodes.device) / 2
    x_eval = (
        midpoints.unsqueeze(1) + 0.5 * cell_widths.unsqueeze(1) * quad_pts
    ).flatten()
    weights = (cell_widths.unsqueeze(1) * weights).flatten()
    return x_eval, weights


def construct_mass_matrix(
    fn1,
    fn2,
    x_nodes,
    eval_types=["single", "single"],
    num_quad_pts=2,
    W=None,
):
    h = x_nodes[1] - x_nodes[0]
    L = x_nodes[-1]
    x_eval, weights = construct_quadrature(x_nodes, num_quad_pts)
    if eval_types[0] == "single":
        fn1_eval = torch.stack([fn1(x_eval, x_c, h, L) for x_c in x_nodes])
        if W is not None:
            fn1_eval = W @ fn1_eval
    else:
        fn1_eval = fn1(x_nodes, x_eval, W=W)
    if eval_types[1] == "single":
        fn2_eval = torch.stack([fn2(x_eval, x_c, h, L) for x_c in x_nodes])
        if W is not None:
            fn2_eval = W @ fn2_eval
    else:
        fn2_eval = fn2(x_nodes, x_eval, W=W)
    mass_matrix = torch.einsum("ik,jk,k->ij", fn1_eval, fn2_eval, weights)
    return mass_matrix


def construct_load_vector(f, x_nodes, num_quad_points=2, W=None):
    h = x_nodes[1] - x_nodes[0]
    L = x_nodes[-1]
    x_eval, weights = construct_quadrature(x_nodes, num_quad_points)
    f_phi_0_evaluated = torch.stack(
        [f(x_eval) * phi_0(x_eval, x_c, h, L) for x_c in x_nodes]
    )
    b = f_phi_0_evaluated @ weights
    if W is not None:
        b = W @ b
    return b


def construct_delta0(num_0forms):
    num_1forms = num_0forms * (num_0forms - 1) // 2
    delta0 = torch.zeros(num_1forms, num_0forms)
    cnt = 0
    for i in range(num_0forms):
        for j in range(i + 1, num_0forms):
            delta0[cnt, i] = -1
            delta0[cnt, j] = 1
            cnt += 1
    return delta0


# the behavior of torch.linalg.lstsq on GPUs is not ideal,
# see https://github.com/pytorch/pytorch/issues/117122
def svd_lstsq(AA, BB, tol=1e-10):
    U, S, Vh = torch.linalg.svd(AA, full_matrices=False, driver="gesvd")
    Spinv = torch.zeros_like(S)
    Spinv[S > tol] = 1 / S[S > tol]
    UhBB = U.adjoint() @ BB
    if Spinv.ndim != UhBB.ndim:
        Spinv = Spinv.unsqueeze(-1)
    SpinvUhBB = Spinv * UhBB
    return Vh.adjoint() @ SpinvUhBB


class NLFlux(nn.Module):
    def __init__(
        self,
        Npou,
        extra_params=0,
        num_hidden_layers=12,
        hidden_layer_width=128,
        activation=nn.Tanh(),
    ):
        super().__init__()
        num_1forms = Npou * (Npou - 1) // 2
        self.activation = activation
        self.tail = nn.Linear(Npou + extra_params, hidden_layer_width)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_layer_width, hidden_layer_width)
                for _ in range(num_hidden_layers)
            ]
        )
        self.head = nn.Linear(hidden_layer_width, num_1forms)

    def forward(self, x):
        x = self.activation(self.tail(x))
        for layer in self.layers:
            x = x + self.activation(layer(x))
        return self.head(x)


class PsiHatGuessMLP(nn.Module):
    def __init__(
        self,
        Npou,
        num_input_params,
        num_hidden_layers=3,
        hidden_layer_width=12,
        activation=nn.Tanh(),
    ):
        super().__init__()
        self.activation = activation
        self.tail = nn.Linear(num_input_params, hidden_layer_width)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_layer_width, hidden_layer_width)
                for _ in range(num_hidden_layers)
            ]
        )
        self.head = nn.Linear(hidden_layer_width, Npou)

    def forward(self, x):
        x = self.activation(self.tail(x))
        for layer in self.layers:
            x = x + self.activation(layer(x))
        return self.head(x)


class SourceMLP(nn.Module):
    def __init__(
        self,
        num_input_params,
        num_output_pous,
        num_hidden_layers=6,
        hidden_layer_width=64,
        activation=nn.Tanh(),
    ):
        super().__init__()
        self.activation = activation
        self.tail = nn.Linear(num_input_params, hidden_layer_width)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_layer_width, hidden_layer_width)
                for _ in range(num_hidden_layers)
            ]
        )
        self.head = nn.Linear(hidden_layer_width, num_output_pous)

        # self.head.weight.data = torch.zeros_like(self.head.weight.data)
        # self.head.bias.data = torch.zeros_like(self.head.bias.data)

    def forward(self, x):
        x = self.activation(self.tail(x))
        for layer in self.layers:
            x = x + self.activation(layer(x))
        return self.head(x)
        # return torch.softmax(self.head(x), dim=-1)
        # return self.head(x).square()
        # return self.head(x).exp()


# NOTE: This module does not handle a leading "batch dimension" yet.
class PairwiseNLFlux(nn.Module):
    def __init__(
        self,
        Npou,
        extra_params=0,
        num_hidden_layers=6,
        hidden_layer_width=32,
        activation=nn.Softplus(),
    ):
        super().__init__()

        self.Npou = Npou
        self.extra_params = extra_params
        self.activation = activation
        self.tail = nn.Linear(2 + extra_params, hidden_layer_width)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_layer_width, hidden_layer_width)
                for _ in range(num_hidden_layers)
            ]
        )
        self.head = nn.Linear(hidden_layer_width, 1)

        self.indexer = torch.tensor(list(itertools.combinations(range(Npou), 2)))

    def forward(self, x):
        # cleave off the extra parameters assumed to be concatenated onto the end of the input vector
        extra_params_vector = x[-self.extra_params :]
        x = x[: -self.extra_params]

        # select the pairwise combinations of all the remaining inputs (psi_hat values)
        x = x[self.indexer]
        # and tack the extra parameters back on
        x = torch.cat([x, extra_params_vector.repeat(x.shape[0], 1)], dim=1)

        x = self.activation(self.tail(x))
        for layer in self.layers:
            x = x + self.activation(layer(x))
        return self.head(x).squeeze()


class PairwiseNLFluxFields(nn.Module):
    def __init__(
        self,
        Npou,
        extra_params=0,
        num_fields=3,
        # num_hidden_layers=12,
        # hidden_layer_width=128,
        num_hidden_layers=12,
        hidden_layer_width=64,
        activation=nn.LeakyReLU(),
    ):
        super().__init__()

        self.Npou = Npou
        self.extra_params = extra_params
        self.num_fields = num_fields
        self.activation = activation
        self.tail = nn.Linear(2 * num_fields + extra_params, hidden_layer_width)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_layer_width, hidden_layer_width)
                for _ in range(num_hidden_layers)
            ]
        )
        self.head = nn.Linear(hidden_layer_width, num_fields)

        self.indexer = torch.tensor(list(itertools.combinations(range(Npou), 2)))

    def single_forward(self, x):
        # cleave off the extra parameters assumed to be concatenated onto the end of the input vector
        extra_params_vector = x[-self.extra_params :]
        x = x[: -self.extra_params]

        # the field values are stacked in the input like
        # (F0_0, F0_1, ..., F0_Npou, F1_0, F1_1, ..., F1_Npou, F2_0, F2_1, ..., F2_Npou, ...);
        # we need to get these into a shape where each row contains all the fields for a given POU
        x = x.reshape(self.num_fields, -1).T

        # select the pairwise combinations of all the remaining inputs (psi_hat values)
        x = x[self.indexer].reshape(-1, 2 * self.num_fields)

        # and tack the extra parameters back on
        x = torch.cat([x, extra_params_vector.repeat(x.shape[0], 1)], dim=1)

        x = self.activation(self.tail(x))
        for layer in self.layers:
            x = x + self.activation(layer(x))
        return self.head(x).squeeze()

    def forward(self, x):
        if len(x.shape) == 1:
            return self.single_forward(x)
        else:
            return torch.vmap(self.single_forward)(x)


class PairwiseNLFluxFields_v2(nn.Module):
    def __init__(
        self,
        Npou,
        extra_params=0,
        num_fields=3,
        num_dimensions=1,
        num_hidden_layers=6,
        hidden_layer_width=64,
        # num_hidden_layers=3,
        # hidden_layer_width=12,
        activation=nn.Tanh(),
        init_gain=None,
    ):
        super().__init__()

        self.Npou = Npou
        self.num_one_forms = num_dimensions * (self.Npou * (self.Npou - 1) // 2)
        self.extra_params = extra_params
        self.num_fields = num_fields
        self.num_dimensions = num_dimensions
        self.activation = activation
        self.tail = nn.Linear(
            2 * num_fields + extra_params + self.num_dimensions, hidden_layer_width
        )
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_layer_width, hidden_layer_width)
                for _ in range(num_hidden_layers)
            ]
        )
        self.head = nn.Linear(hidden_layer_width, num_fields)
        self.head.weight.data = torch.zeros_like(self.head.weight.data)
        self.head.bias.data = torch.zeros_like(self.head.bias.data)

        self.indexer = torch.tensor(list(itertools.combinations(range(Npou), 2)))

        if init_gain is not None:
            nn.init.xavier_normal_(self.tail.weight, gain=init_gain)
            nn.init.zeros_(self.tail.bias)
            for layer in self.layers:
                nn.init.xavier_normal_(layer.weight, gain=init_gain)
                nn.init.zeros_(layer.bias)
            nn.init.xavier_normal_(self.head.weight, gain=init_gain)
            nn.init.zeros_(self.head.bias)

    def single_forward(self, x, flip=False):
        # cleave off the extra parameters assumed to be concatenated onto the end of the input vector
        extra_params_vector = x[
            -(self.extra_params + self.num_one_forms) : -self.num_one_forms
        ]
        oriented_areas = x[-self.num_one_forms :]
        x = x[: -(self.extra_params + self.num_one_forms)]

        # the field values are stacked in the input like
        # (F0_0, F0_1, ..., F0_Npou, F1_0, F1_1, ..., F1_Npou, F2_0, F2_1, ..., F2_Npou, ...);
        # we need to get these into a shape where each row contains all the fields for a given POU
        x = x.reshape(self.num_fields, -1).T

        # select the pairwise combinations of all the remaining inputs (psi_hat values)
        if flip:
            x = x[self.indexer.flip(1)].reshape(-1, 2 * self.num_fields)
        else:
            x = x[self.indexer].reshape(-1, 2 * self.num_fields)

        # this ^ is:
        # (F0_i, F1_i, F2_i, F0_j, F1_j, F2_j) for each pair (i, j), i < j of the POU indices
        # this is where I want to tack on the oriented pairwise area
        if self.num_dimensions == 1:
            x = torch.cat([x, oriented_areas.unsqueeze(-1)], dim=1)
        else:
            x = torch.cat([x, oriented_areas.reshape(-1, self.num_dimensions)], dim=1)

        # and tack the extra parameters back on
        x = torch.cat([x, extra_params_vector.repeat(x.shape[0], 1)], dim=1)

        x = self.activation(self.tail(x))
        for layer in self.layers:
            x = x + self.activation(layer(x))
        return self.head(x).squeeze()

    def get_weights_and_biases_sum(self):
        return sum(
            [
                torch.sum(layer.weight.abs()) + torch.sum(layer.bias.abs())
                for layer in [self.tail] + list(self.layers) + [self.head]
            ]
        )

    def forward(self, x):
        if len(x.shape) == 1:
            return self.single_forward(x) + self.single_forward(x, flip=True)
        else:
            return torch.vmap(
                lambda x: self.single_forward(x) + self.single_forward(x, flip=True)
            )(x)


class W_NN(nn.Module):
    def __init__(
        self,
        Npou,  # desired number of POUs *not* including boundary POUs
        num_nodes,  # number of nodes on the fine mesh
        num_input_params,
        boundary_mask=None,  # tensor the same shape as the fine mesh; with True if the node is a dirichlet boundary node and False otherwise
        boundary_values=None,  # tensor the same shape as the fine mesh; with boundary values set wherever the above tensor is True
        num_hidden_layers=6,
        hidden_layer_width=64,
        activation=nn.Tanh(),
        reverse_sort_bc_list=False,  # this is a hack to make the boundary values in the correct order for the turbulence problem
    ):
        super().__init__()

        self.Npou = Npou
        self.activation = activation

        self.tail = nn.Linear(num_input_params, hidden_layer_width)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_layer_width, hidden_layer_width)
                for _ in range(num_hidden_layers)
            ]
        )
        self.head = nn.Linear(hidden_layer_width, Npou * num_nodes)
        self.output_shape = (Npou, num_nodes)

        self.boundary_mask = boundary_mask
        if boundary_mask is not None:
            self.W_mask = boundary_mask.flatten().repeat(
                Npou, 1
            )  # W_ij == True if j is a boundary node
            self.unique_boundary_values = torch.unique(boundary_values[boundary_mask])
            if reverse_sort_bc_list:
                self.unique_boundary_values = torch.flip(
                    self.unique_boundary_values, [0]
                )
            bc_node_indices_list = []
            for bc_value in self.unique_boundary_values:
                indices = (
                    (boundary_mask.flatten() & (boundary_values.flatten() == bc_value))
                    .nonzero()
                    .squeeze()
                )
                bc_node_indices_list.append(indices)
            self.bc_pous_add_on = torch.zeros(
                self.unique_boundary_values.shape[0],
                num_nodes,
                dtype=torch.float64,
                device=boundary_mask.device,
            )
            for i, bc_node_indices in enumerate(bc_node_indices_list):
                self.bc_pous_add_on[i, bc_node_indices] = 1.0
            self.non_boundary_node_indices = (~self.W_mask[0]).nonzero().squeeze()
        self.actual_Npou = Npou + (
            self.bc_pous_add_on.shape[0] if self.boundary_mask is not None else 0
        )

    def single_forward(self, x):
        x = self.activation(self.tail(x))
        for layer in self.layers:
            x = x + self.activation(layer(x))
        W = self.head(x).reshape(self.output_shape)

        if self.boundary_mask is None:
            return torch.softmax(W, dim=0)
        else:
            shortened_W = W[~self.W_mask].reshape(self.Npou, -1)
            shortened_W = torch.softmax(shortened_W, dim=0)
            full_W = torch.zeros_like(W)
            full_W[:, self.non_boundary_node_indices] = shortened_W
            # W = torch.where(
            #     self.W_mask, -torch.inf, W
            # )  # this ensures nodes that are on the boundary do not get assigned to a POU in the next step
            # W = torch.softmax(W, dim=0).nan_to_num()
            # return torch.cat(
            #     [W, self.bc_pous_add_on], dim=0
            # )  # tack on the boundary POUs as defined above
            return torch.cat(
                [full_W, self.bc_pous_add_on], dim=0
            )  # tack on the boundary POUs as defined above

    def forward(self, x):
        if len(x.shape) == 1:
            return self.single_forward(x)
        else:
            return torch.vmap(self.single_forward)(x)


class W_NN_v2(nn.Module):
    def __init__(
        self,
        Npou,  # desired number of POUs *not* including boundary POUs
        nodes,  # the nodes of the fine mesh, supplied by WhitneyFormsND class "nodes" attribute
        num_input_params,
        boundary_mask=None,  # tensor the same shape as the fine mesh; with True if the node is a dirichlet boundary node and False otherwise
        boundary_values=None,  # tensor the same shape as the fine mesh; with boundary values set wherever the above tensor is True
        num_hidden_layers=12,
        hidden_layer_width=128,
        activation=nn.Tanh(),
        use_box_init=False,
        noise_level=3.0,
    ):
        super().__init__()

        self.Npou = Npou
        self.noise_level = noise_level

        # normalize the entries in nodes independently for each dimensions
        min = nodes.min(dim=0).values
        width = nodes.max(dim=0).values - min
        self.nodes = (nodes - min) / width
        self.nodes = nodes

        num_nodes = nodes.shape[0]
        num_dimensions = nodes.shape[1]

        self.activation = activation
        self.tail = nn.Linear(num_dimensions + num_input_params, hidden_layer_width)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_layer_width, hidden_layer_width)
                for _ in range(num_hidden_layers)
            ]
        )
        self.head = nn.Linear(hidden_layer_width, Npou)
        self.output_shape = (Npou, num_nodes)
        self.boundary_mask = boundary_mask
        if boundary_mask is not None:
            self.W_mask = boundary_mask.flatten().repeat(
                Npou, 1
            )  # W_ij == True if j is a boundary node
            self.unique_boundary_values = torch.unique(boundary_values[boundary_mask])
            bc_node_indices_list = []
            for bc_value in self.unique_boundary_values:
                indices = (
                    (boundary_mask.flatten() & (boundary_values.flatten() == bc_value))
                    .nonzero()
                    .squeeze()
                )
                bc_node_indices_list.append(indices)
            self.bc_pous_add_on = torch.zeros(
                self.unique_boundary_values.shape[0],
                num_nodes,
                dtype=torch.float64,
                device=boundary_mask.device,
            )
            for i, bc_node_indices in enumerate(bc_node_indices_list):
                self.bc_pous_add_on[i, bc_node_indices] = 1.0
            self.non_boundary_node_indices = (~self.W_mask[0]).nonzero().squeeze()
        self.actual_Npou = Npou + (
            self.bc_pous_add_on.shape[0] if self.boundary_mask is not None else 0
        )

        if use_box_init:
            box_init_resnet(self.tail, 1, len(self.layers) + 2)
            for l, layer in enumerate(self.layers[:-1]):
                box_init_resnet(layer, l + 2, len(self.layers) + 2)
            box_init_resnet(self.head, len(self.layers) + 2, len(self.layers) + 2)

    def reduce_noise(self, frac):
        self.noise_level *= frac

    def single_forward(self, x):
        x = torch.cat([self.nodes, x.repeat(self.nodes.shape[0], 1)], dim=-1)
        x = self.activation(self.tail(x))
        for i, layer in enumerate(self.layers):
            x = x + self.activation(layer(x))
        x = self.head(x)
        x = x.permute(1, 0)
        # x = x + self.noise_level * torch.randn_like(x)
        # x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
        W = x
        if self.boundary_mask is None:
            return torch.softmax(W, dim=0)
        else:
            shortened_W = W[~self.W_mask].reshape(self.Npou, -1)
            shortened_W = torch.softmax(shortened_W, dim=0)
            full_W = torch.zeros_like(W)
            full_W[:, self.non_boundary_node_indices] = shortened_W
            # W = torch.where(
            #     self.W_mask, -torch.inf, W
            # )  # this ensures nodes that are on the boundary do not get assigned to a POU in the next step
            # W = torch.softmax(W, dim=0).nan_to_num()
            # return torch.cat(
            #    [W, self.bc_pous_add_on], dim=0
            # )  # tack on the boundary POUs as defined above
            return torch.cat(
                [full_W, self.bc_pous_add_on], dim=0
            )  # tack on the boundary POUs as defined above

    def forward(self, x):
        if len(x.shape) == 1:
            return self.single_forward(x)
        else:
            return torch.vmap(self.single_forward, randomness="different")(x)


class W_GNN(torch.nn.Module):
    def __init__(
        self,
        Npou,
        num_node_features,
        num_graph_level_input_features,
        boundary_mask=None,
        boundary_values=None,
        num_hidden_layers=12,
        hidden_channels=64,
        activation=nn.Tanh(),
    ):
        super().__init__()

        self.Npou = Npou
        self.activation = activation

        self.tail = gnn.GCNConv(
            num_node_features + num_graph_level_input_features, hidden_channels
        )
        self.layers = torch.nn.ModuleList(
            [
                gnn.GCNConv(hidden_channels, hidden_channels)
                for _ in range(num_hidden_layers)
            ]
        )
        self.head = gnn.GCNConv(hidden_channels, Npou)

        self.boundary_mask = boundary_mask
        if boundary_mask is not None:
            self.W_mask = boundary_mask.flatten().repeat(
                Npou, 1
            )  # W_ij == True if j is a boundary node
            self.unique_boundary_values = torch.unique(boundary_values[boundary_mask])
            bc_node_indices_list = []
            for bc_value in self.unique_boundary_values:
                indices = (
                    (boundary_mask.flatten() & (boundary_values.flatten() == bc_value))
                    .nonzero()
                    .squeeze()
                )
                bc_node_indices_list.append(indices)
            self.bc_pous_add_on = torch.zeros(
                self.unique_boundary_values.shape[0],
                torch.numel(boundary_mask),
                dtype=torch.float64,
                device=boundary_mask.device,
            )
            for i, bc_node_indices in enumerate(bc_node_indices_list):
                self.bc_pous_add_on[i, bc_node_indices] = 1.0
            self.non_boundary_node_indices = (~self.W_mask[0]).nonzero().squeeze()
        self.actual_Npou = Npou + (
            self.bc_pous_add_on.shape[0] if self.boundary_mask is not None else 0
        )

    def single_forward(self, data, z):
        x, edge_index = data.x, data.edge_index

        z_projected = z.unsqueeze(0).expand(x.size(0), -1)
        x = torch.cat([x, z_projected], dim=-1)
        x = self.activation(self.tail(x, edge_index))
        for layer in self.layers:
            x = self.activation(layer(x, edge_index))
        W = self.head(x, edge_index).T

        if self.boundary_mask is None:
            return torch.softmax(W, dim=0)
        else:
            shortened_W = W[~self.W_mask].reshape(self.Npou, -1)
            shortened_W = torch.softmax(shortened_W, dim=0)
            full_W = torch.zeros_like(W)
            full_W[:, self.non_boundary_node_indices] = shortened_W
            return torch.cat(
                [full_W, self.bc_pous_add_on], dim=0
            )  # tack on the boundary POUs as defined above

    def forward(self, data, z):
        if len(z.shape) == 1:
            return self.single_forward(data, z)
        else:
            return torch.vmap(lambda z: self.single_forward(data, z))(z)


class W_Unet_GNN(torch.nn.Module):
    def __init__(
        self,
        Npou,
        num_node_features,
        num_graph_level_input_features,
        boundary_mask=None,
        boundary_values=None,
        num_hidden_layers=12,
        hidden_channels=64,
        activation=nn.functional.relu,
    ):
        super().__init__()

        self.Npou = Npou
        self.activation = activation

        self.unet = gnn.models.GraphUNet(
            num_node_features + num_graph_level_input_features,
            hidden_channels,
            Npou,
            num_hidden_layers,
            act=activation,
        )

        self.graph_pool = gnn.global_mean_pool
        self.graph_output = nn.Linear(hidden_channels, Npou)

        self.boundary_mask = boundary_mask
        if boundary_mask is not None:
            self.W_mask = boundary_mask.flatten().repeat(
                Npou, 1
            )  # W_ij == True if j is a boundary node
            self.unique_boundary_values = torch.unique(boundary_values[boundary_mask])
            bc_node_indices_list = []
            for bc_value in self.unique_boundary_values:
                indices = (
                    (boundary_mask.flatten() & (boundary_values.flatten() == bc_value))
                    .nonzero()
                    .squeeze()
                )
                bc_node_indices_list.append(indices)
            self.bc_pous_add_on = torch.zeros(
                self.unique_boundary_values.shape[0],
                torch.numel(boundary_mask),
                dtype=torch.float64,
                device=boundary_mask.device,
            )
            for i, bc_node_indices in enumerate(bc_node_indices_list):
                self.bc_pous_add_on[i, bc_node_indices] = 1.0
            self.non_boundary_node_indices = (~self.W_mask[0]).nonzero().squeeze()
        self.actual_Npou = Npou + (
            self.bc_pous_add_on.shape[0] if self.boundary_mask is not None else 0
        )

    def single_forward(self, data, z):
        x, edge_index = data.x, data.edge_index

        # z_projected = z.unsqueeze(0).expand(x.size(0), -1)
        z_projected = z.repeat(x.shape[0], 1)
        x = torch.cat([x, z_projected], dim=-1).contiguous()
        x = self.unet(x, edge_index, None)
        W = x.T

        # graph_level_x = self.graph_pool(x, None)
        # graph_level_output = self.graph_output(graph_level_x).squeeze().square()

        if self.boundary_mask is None:
            return torch.softmax(W, dim=0)
        else:
            shortened_W = W[~self.W_mask].reshape(self.Npou, -1)
            shortened_W = torch.softmax(shortened_W, dim=0)
            full_W = torch.zeros_like(W)
            full_W[:, self.non_boundary_node_indices] = shortened_W
            return torch.cat([full_W, self.bc_pous_add_on], dim=0)
            # tack on the boundary POUs as defined above

    def forward(self, data, z):
        if len(z.shape) == 1:
            return self.single_forward(data, z)
        else:
            # return torch.vmap(lambda z: self.single_forward(data, z))(z)
            return torch.stack(
                [self.single_forward(data, z[i]) for i in range(z.shape[0])]
            )


def analytic_sod_soln(eval_points, Ul, Ur, gamma):
    rhol, jl, El = Ul
    rhor, jr, Er = Ur

    ul = jl / rhol
    ur = jr / rhor
    pl = (gamma - 1.0) * (El - 0.5 * rhol * ul**2)
    pr = (gamma - 1.0) * (Er - 0.5 * rhor * ur**2)

    cl = np.sqrt(gamma * pl / rhol)
    cr = np.sqrt(gamma * pr / rhor)

    def phil(p):
        if p <= pl:
            u = ul + 2 * cl / (gamma - 1) * (
                1.0 - (p / pl) ** ((gamma - 1) / (2 * gamma))
            )
        else:
            u = ul + (
                2
                * cl
                / np.sqrt(2 * gamma * (gamma - 1))
                * (1.0 - p / pl)
                / np.sqrt(1 + p / pl * (gamma + 1) / (gamma - 1))
            )
        return u

    def phir(p):
        if p <= pr:
            u = ur - 2 * cr / (gamma - 1) * (
                1.0 - (p / pr) ** ((gamma - 1) / (2 * gamma))
            )
        else:
            u = ur - (
                2
                * cr
                / np.sqrt(2 * gamma * (gamma - 1))
                * (1.0 - p / pr)
                / np.sqrt(1 + p / pr * (gamma + 1) / (gamma - 1))
            )
        return u

    def phi(p):
        return phir(p) - phil(p)

    ps = scipy.optimize.root_scalar(phi, x0=pr, x1=pl).root
    us = phil(ps)
    rhosr = (
        (1 + (gamma + 1) / (gamma - 1) * ps / pr)
        / (ps / pr + (gamma + 1) / (gamma - 1))
        * rhor
    )
    rhosl = (ps / pl) ** (1.0 / gamma) * rhol

    s = (rhosr * us - rhor * ur) / (rhosr - rhor)

    csl = np.sqrt(gamma * ps / rhosl)
    laml = ul - cl
    lamsl = us - csl

    xtr = torch.reshape(eval_points, (-1, 2))

    u = torch.zeros(xtr.shape[0], 3)

    for i, xti in enumerate(xtr):
        if xti[0] < 1e-12:
            if xti[1] < 0:
                rho = rhol
                v = ul
                p = pl
            else:
                rho = rhor
                v = ur
                p = pr
        else:
            xi = xti[1] / xti[0]
            if xi <= laml:
                rho = rhol
                v = ul
                p = pl
            elif xi > laml and xi <= lamsl:
                v = ((gamma - 1) * ul + 2 * (cl + xi)) / (gamma + 1)
                rho = (rhol**gamma * (v - xi) ** 2 / (gamma * pl)) ** (
                    1.0 / (gamma - 1)
                )
                p = (pl / rhol**gamma) * rho**gamma
            elif xi > lamsl and xi <= us:
                rho = rhosl
                v = us
                p = ps
            elif xi > us and xi <= s:
                rho = rhosr
                v = us
                p = ps
            else:
                rho = rhor
                v = ur
                p = pr

        u[i, 0] = rho
        u[i, 1] = rho * v
        u[i, 2] = p / (gamma - 1) + 0.5 * rho * v**2

    u = u.reshape(eval_points.shape[:-1] + (3,))

    return u
