import torch
import random
import numpy as np


class LowRankProjector:
    def __init__(
            self, rank, scale, proj_type,
            st_init_step_size, subspace_update_method,
            st_step_size_scheduler, subspace_update_interval,
            hybrid=False, hybrid_prob=0.3,
            random_rank_r=False, update_rank=1,
            tangent_energy=False, energy_thresh=None,
            random_init=False, random_projection=False,
            verbose=False, step_size_tuning=False,
            log_grad=False, module_name=None,
            no_qr=False
    ):
        self.rank = rank
        self.verbose = verbose
        self.scale = scale
        self.ortho_matrix = None
        self.prev_ortho_matrix = None
        self.proj_type = proj_type
        self.subspace_update_method = subspace_update_method

        self.st_init_step_size = st_init_step_size
        self.st_step_size = st_init_step_size
        self.st_step_size_scheduler = st_step_size_scheduler
        self.subspace_update_interval = subspace_update_interval

        self.hybrid = hybrid
        self.hybrid_prob = hybrid_prob

        self.random_rank_r = random_rank_r
        self.update_rank = update_rank
        self.tangent_energy = tangent_energy
        self.energy_thresh = energy_thresh
        self.random_init = random_init

        self.random_projection = random_projection

        self.step_size_tuning = step_size_tuning
        self.log_grad = log_grad
        self.module_name = module_name

        self.no_qr = no_qr

    def project(self, full_rank_grad, iter):
        if self.subspace_update_method == 'galore':
            low_rank_grad = self.galore_projector(full_rank_grad, iter)
        elif self.subspace_update_method == 'subtrack':
            low_rank_grad = self.subtrack_projector(full_rank_grad, iter)
        else:
            raise ValueError('method should be galore or subtrack')

        return low_rank_grad

    def galore_projector(self, full_rank_grad, iter):
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.subspace_update_interval == 0:
                    if self.ortho_matrix is not None:
                        self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type='right', random_projection=(self.random_init and (iter == 0)))
                    if self.prev_ortho_matrix is None:
                        self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter % self.subspace_update_interval == 0:
                    if self.ortho_matrix is not None:
                        self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type='left', random_projection=(self.random_init and (iter == 0)))
                    if self.prev_ortho_matrix is None:
                        self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.subspace_update_interval == 0:
                    if self.ortho_matrix is not None:
                        self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type='left', random_projection=(self.random_init and (iter == 0)))
                    if self.prev_ortho_matrix is None:
                        self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.subspace_update_interval == 0:
                    if self.ortho_matrix is not None:
                        self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type='right', random_projection=(self.random_init and (iter == 0)))
                    if self.prev_ortho_matrix is None:
                        self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.subspace_update_interval == 0:
                if self.ortho_matrix is not None:
                    self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type='right', random_projection=(self.random_init and (iter == 0)))
                if self.prev_ortho_matrix is None:
                    self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.subspace_update_interval == 0:
                if self.ortho_matrix is not None:
                        self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type='left', random_projection=(self.random_init and (iter == 0)))
                if self.prev_ortho_matrix is None:
                    self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.subspace_update_interval == 0:
                if self.ortho_matrix is not None:
                    prev_0 = self.ortho_matrix[0].clone().detach()
                    prev_1 = self.ortho_matrix[1].clone().detach()
                    self.prev_ortho_matrix = [prev_0, prev_1]
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type='full', random_projection=(self.random_init and (iter == 0)))
                if self.prev_ortho_matrix is None:
                    self.prev_ortho_matrix = [prev_0, prev_1]
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()

        return low_rank_grad

    def set_adjustment_method(self):
        if self.hybrid:
            p = random.random()
            if p <= self.hybrid_prob:
                print("\n======= Adjustment: GrassJump =======\n")
                self.random_rank_r = False
                self.random_init = True
                self.random_projection = True
            else:
                print("\n======= Adjustment: GrassWalk =======\n")
                self.random_rank_r = True
                self.random_init = False
                self.random_projection = False

    def subtrack_projector(self, full_rank_grad, iter):
        if iter == 0:
            self.random_init = False if self.hybrid else self.random_init
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                self.ortho_matrix = self.get_orthogonal_matrix(
                     full_rank_grad, self.rank, type='right', random_projection=(self.random_projection or self.random_init))
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
                if self.ortho_matrix is not None:
                    self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
            else:
                self.ortho_matrix = self.get_orthogonal_matrix(
                     full_rank_grad, self.rank, type='left', random_projection=(self.random_projection or self.random_init))
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
                if self.ortho_matrix is not None:
                    self.prev_ortho_matrix = self.ortho_matrix.clone().detach()

        elif (iter % self.subspace_update_interval) != 0:

            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)

        else:
            self.set_adjustment_method()
            self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
            
            if self.random_projection:
                if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right',
                                                                   random_projection=self.random_projection)
                else:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left',
                                                                   random_projection=self.random_projection)
            else:
                if self.st_step_size_scheduler == "iterative_decrease":
                    self.st_step_size = self.st_init_step_size/iter

                self.track_the_subspace(full_rank_grad)

            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)

        return low_rank_grad

    def track_the_subspace(self, full_rank_grad):
        if self.ortho_matrix.dtype != torch.float:
            float_data = False
            original_type = self.ortho_matrix.dtype
            full_rank_grad = full_rank_grad.float()
            self.ortho_matrix = self.ortho_matrix.float()
        else:
            float_data = True

        if (not (self.random_rank_r or self.random_projection)):

            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                estimated_w = torch.linalg.lstsq(
                    self.ortho_matrix.t(), full_rank_grad.t()
                ).solution.t()
                residual = full_rank_grad - torch.matmul(estimated_w, self.ortho_matrix)
                partial_derivative = -2 * torch.matmul(estimated_w.t(), residual)
                tangent_vector = torch.matmul(
                    partial_derivative,
                    (torch.eye(self.ortho_matrix.shape[1]).to('cuda') - torch.matmul(self.ortho_matrix.t(),
                                                                                     self.ortho_matrix))
                )
            else:
                estimated_w = torch.linalg.lstsq(
                    self.ortho_matrix, full_rank_grad
                ).solution
                residual = full_rank_grad - torch.matmul(self.ortho_matrix, estimated_w)
                partial_derivative = -2 * torch.matmul(residual, estimated_w.t())

                tangent_vector = torch.matmul(
                    (torch.eye(self.ortho_matrix.shape[0]).to('cuda') - torch.matmul(self.ortho_matrix, self.ortho_matrix.t())),
                    partial_derivative
                )

        if self.random_rank_r:
            random_vector = torch.randn_like(self.ortho_matrix)
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                random_vector = random_vector - torch.matmul(
                    torch.matmul(random_vector, self.ortho_matrix.t()), self.ortho_matrix
                )
            else:
                random_vector = random_vector - torch.matmul(
                    self.ortho_matrix, torch.matmul(self.ortho_matrix.t(), random_vector)
                )
            if not (self.energy_thresh is None):
                random_vector = random_vector / (torch.norm(random_vector, p='fro') + 1e-10) * self.energy_thresh
            tangent_vector = random_vector

        if self.random_rank_r:
            U, Sigma, V = self.random_rank_k_matrix_estimation(tangent_vector, k=self.update_rank)
        else:
            U, Sigma, V = self.rank_k_matrix_estimation(tangent_vector, k=self.update_rank)

        if self.step_size_tuning:
            Sigma = Sigma/torch.norm(Sigma, p='fro')
            
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            self.ortho_matrix = torch.matmul(
                U, torch.matmul(
                    torch.concat([torch.cos(self.st_step_size * Sigma), torch.sin(-1 * self.st_step_size * Sigma)], 0),
                    torch.concat([torch.matmul(U.t(), self.ortho_matrix), V.t()], 0)
                ).reshape(Sigma.shape[0], self.ortho_matrix.shape[1])
            ) + torch.matmul(
                (torch.eye(U.shape[0]).to("cuda") - torch.matmul(U, U.t())), self.ortho_matrix
            )
        else:
            self.ortho_matrix = torch.matmul(
                torch.matmul(
                    torch.concat([torch.matmul(self.ortho_matrix, V), U], 1),
                    torch.concat([torch.cos(self.st_step_size * Sigma), torch.sin(-1 * self.st_step_size * Sigma)], 0)
                ).reshape((self.ortho_matrix.shape[0]), Sigma.shape[0]), V.t()
            ) + torch.matmul(
                self.ortho_matrix, (torch.eye(V.shape[0]).to("cuda") - torch.matmul(V, V.t()))
            )

        if not float_data:
            self.ortho_matrix = self.ortho_matrix.to(original_type)

    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]

        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type, random_projection):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        if not random_projection:
                U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)
                if self.log_grad:
                    with open(f'grads/grad_singular_values_{self.module_name}.txt', 'a+') as f:
                        f.write(str(s) + '\n')

                # make the smaller matrix always to be orthogonal matrix
                if type == 'right':
                    A = U[:, :rank] @ torch.diag(s[:rank])
                    B = Vh[:rank, :]

                    if not float_data:
                        B = B.to(original_device).type(original_type)
                    return B
                elif type == 'left':
                    A = U[:, :rank]
                    B = torch.diag(s[:rank]) @ Vh[:rank, :]
                    if not float_data:
                        A = A.to(original_device).type(original_type)
                    return A
                elif type == 'full':
                    A = U[:, :rank]
                    B = Vh[:rank, :]
                    if not float_data:
                        A = A.to(original_device).type(original_type)
                        B = B.to(original_device).type(original_type)
                    return [A, B]
                else:
                    raise ValueError('type should be left, right or full')
        else:
            if self.no_qr:
                if type == 'right':
                    B = self.get_random_normalized_matrix(rank, weights.shape[1])
                    if not float_data:
                        B = B.to(original_device).type(original_type)
                    return B
                elif type == 'left':
                    A = self.get_random_normalized_matrix(weights.shape[0], rank)
                    if not float_data:
                        A = A.to(original_device).type(original_type)
                    return A
            else:
                if type == 'right':
                    B = self.get_random_orthogonal_matrix(rank, weights.shape[1])
                    if not float_data:
                        B = B.to(original_device).type(original_type)
                    return B
                elif type == 'left':
                    A = self.get_random_orthogonal_matrix(weights.shape[0], rank)
                    if not float_data:
                        A = A.to(original_device).type(original_type)
                    return A

    @torch.no_grad()
    def get_random_orthogonal_matrix(self, n, m):
        Z = torch.randn(n, m)
        if n >= m:
            # Columns orthonormal: Q^T Q = I_m, shape (n, m)
            Q, _ = torch.linalg.qr(Z, mode="reduced")
            return Q
        else:
            # Rows orthonormal: (Q^T) (Q) = I_n after transpose
            Qt, _ = torch.linalg.qr(Z.T, mode="reduced")  # (m, n), columns orthonormal
            return Qt.T  # (n, m)

    @torch.no_grad()
    def get_random_normalized_matrix(self, n, m):
        Z = torch.randn(n, m)
        if n >= m:
            # Normalize columns (so each column has unit length)
            Z = Z / (Z.norm(dim=0, keepdim=True) + 1e-8)
        else:
            # Normalize rows
            Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-8)
        return Z

    def rank_k_matrix_estimation(self, matrix, k=1):

        U, Sigma, Vt = torch.linalg.svd(matrix, full_matrices=False)

        if k == 1:
            return U[:, :k], Sigma[:k], Vt.t()[:, :k]
        return U[:, :k], torch.diag(Sigma[:k]), Vt.t()[:, :k]

    def random_rank_k_matrix_estimation(self, matrix, k=1, oversample=8, n_power_iter=0, rng=None):
        m, n = matrix.shape
        l = min(k + oversample, min(m, n))

        device, dtype = matrix.device, matrix.dtype
        Omega = torch.randn(n, l, device=device, dtype=dtype, generator=rng)

        # Sketch & (optional) power iterations
        Y = matrix @ Omega  # (m, l)
        for _ in range(max(0, n_power_iter)):
            Y = matrix @ (matrix.transpose(-2, -1) @ Y)  # (m, l)

        # Orthonormal basis for the range of A
        Q, _ = torch.linalg.qr(Y, mode='reduced')  # (m, l)

        # Project to small matrix and do exact SVD there
        B = Q.transpose(-2, -1) @ matrix  # (l, n)
        Ub, S, Vt = torch.linalg.svd(B, full_matrices=False)  # Ub: (l, l), S: (l,), Vt: (n, l)

        # Lift left singular vectors back to R^m
        U = Q @ Ub  # (m, l)
        V = Vt.transpose(-2, -1)  # (n, l)

        U_k = U[:, :k]
        S_k = S[:k]
        V_k = V[:, :k]

        if k == 1:
            return U_k, S_k, V_k
        return U_k, torch.diag(S_k), V_k
