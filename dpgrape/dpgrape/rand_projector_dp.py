# Random Projector class for DP-GRAPE, based off of galore_torch.galore_projector.py

import torch

class RandProjectorDP:
    def __init__(self, rank, rand_type, verbose=False, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.scale = scale
        self.proj_matrix = None   
        self.proj_type = proj_type
        self.rand_type = rand_type
        self.seed = None

    def generate(self, full_rank_grad_shape, grad_dtype, grad_device):
        if self.proj_type == 'std':
            if full_rank_grad_shape[0] >= full_rank_grad_shape[1]:
                self.proj_matrix = self.get_projection_matrix(full_rank_grad_shape, grad_dtype, grad_device, self.rank, type='right')
            else:
                self.proj_matrix = self.get_projection_matrix(full_rank_grad_shape, grad_dtype, grad_device, self.rank, type='left')
        elif self.proj_type == 'reverse_std':
            if full_rank_grad_shape[0] >= full_rank_grad_shape[1]:
                self.proj_matrix = self.get_projection_matrix(full_rank_grad_shape, grad_dtype, grad_device, self.rank, type='left')
            else:
                self.proj_matrix = self.get_projection_matrix(full_rank_grad_shape, grad_dtype, grad_device, self.rank, type='right')
        elif self.proj_type == 'right':
                self.proj_matrix = self.get_projection_matrix(full_rank_grad_shape, grad_dtype, grad_device, self.rank, type='right')
        elif self.proj_type == 'left':
                self.proj_matrix = self.get_projection_matrix(full_rank_grad_shape, grad_dtype, grad_device, self.rank, type='left')
        elif self.proj_type == 'full':
                self.proj_matrix = self.get_projection_matrix(full_rank_grad_shape, grad_dtype, grad_device, self.rank, type='full')


    def project(self, full_rank_grad, batch=True):
        if batch:
            if self.proj_type == 'std':
                if full_rank_grad.shape[1] >= full_rank_grad.shape[2]:
                    low_rank_grad = torch.matmul(full_rank_grad, self.proj_matrix.t())
                else:
                    low_rank_grad = torch.matmul(self.proj_matrix.t(), full_rank_grad)
            elif self.proj_type == 'reverse_std':
                if full_rank_grad.shape[1] >= full_rank_grad.shape[2]:
                    low_rank_grad = torch.matmul(self.proj_matrix.t(),full_rank_grad)
                else:
                    low_rank_grad = torch.matmul(full_rank_grad,self.proj_matrix.t())
            elif self.proj_type == 'right':
                low_rank_grad = torch.matmul(full_rank_grad, self.proj_matrix.t())
            elif self.proj_type == 'left':
                low_rank_grad = torch.matmul(self.proj_matrix.t(), full_rank_grad)
            elif self.proj_type == 'full':
                low_rank_grad = torch.matmul(self.proj_matrix[0].t(), full_rank_grad) @ self.proj_matrix[1].t()
        else:
            if self.proj_type == 'std':
                if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                    low_rank_grad = torch.matmul(full_rank_grad, self.proj_matrix.t())
                else:
                    low_rank_grad = torch.matmul(self.proj_matrix.t(), full_rank_grad)
            elif self.proj_type == 'reverse_std':
                if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                    low_rank_grad = torch.matmul(self.proj_matrix.t(), full_rank_grad)
                else:
                    low_rank_grad = torch.matmul(full_rank_grad,self.proj_matrix.t())
            elif self.proj_type == 'right':
                low_rank_grad = torch.matmul(full_rank_grad, self.proj_matrix.t())
            elif self.proj_type == 'left':
                low_rank_grad = torch.matmul(self.proj_matrix.t(), full_rank_grad)
            elif self.proj_type == 'full':
                low_rank_grad = torch.matmul(self.proj_matrix[0].t(), full_rank_grad) @ self.proj_matrix[1].t()

        return low_rank_grad


    def project_back(self, low_rank_grad):

        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                if self.proj_matrix.shape[0] >= self.proj_matrix.shape[1]:
                    full_rank_grad = torch.matmul(low_rank_grad, self.proj_matrix.t())
                else:
                    full_rank_grad = torch.matmul(low_rank_grad, self.proj_matrix)
            else:
                full_rank_grad = torch.matmul(self.proj_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.proj_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.proj_matrix)
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.proj_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.proj_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.proj_matrix[0], low_rank_grad) @ self.proj_matrix[1]


        return full_rank_grad * self.scale


    # Random projection matrix
    def get_projection_matrix(self, weights_shape, weights_dtype, weights_device, rank, type):

        if weights_dtype != torch.float:
            float_data = False
            original_type = weights_dtype
            original_device = weights_device
        else:
            float_data = True

        # Get seeded generator
        generator = torch.Generator(device=weights_device).manual_seed(self.seed)

        if self.rand_type.lower() == 'gaussian':
            if type == 'right':
                random_mat = torch.normal(mean=0, std=1/rank, size=(rank, weights_shape[1]), generator=generator, device=generator.device, dtype=weights_dtype)
            elif type == 'left':
                random_mat = torch.normal(mean=0, std=1/rank, size=(weights_shape[0], rank), generator=generator, device=generator.device, dtype=weights_dtype)
            return random_mat
        elif self.rand_type.lower() == 'orthonormal':  # Not using in experiments
            U, _, Vh = torch.linalg.svd(torch.normal(mean=0, std=1, size=weights_shape, generator=generator, device=generator.device), full_matrices = False)
            if type=='right':
                B = Vh[:rank, :]
                if not float_data:
                    B = B.to(original_device).type(original_type)
                return B
            elif type=='left':
                A = U[:, :rank]
                if not float_data:
                    A = A.to(original_device).type(original_type)
                return A
            elif type=='full':
                A = U[:, :rank]
                B = Vh[:rank, :]
                if not float_data:
                    A = A.to(original_device).type(original_type)
                    B = B.to(original_device).type(original_type)
                return [A, B]
            else:
                raise ValueError('type should be left, right or full')
        else:
            raise ValueError('type should be orthonormal or gaussian')
        
    # Delete projection matrix
    def clear_projection_matrix(self):
        self.proj_matrix = None

    # Update random seed
    def update_seed(self):
        self.seed = int(torch.randint(low=0, high=2**32, size=(1,)))