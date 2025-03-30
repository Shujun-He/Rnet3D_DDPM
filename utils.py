import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import numpy as np
import yaml

def visualize_point_cloud_batch(batch_xyz, title="Batch Point Clouds"):
    """
    Visualize a batch of point clouds using Plotly.
    
    Args:
        batch_xyz (torch.Tensor): A tensor of shape (bs, l, 3) containing XYZ point clouds.
        title (str): Plot title.
    """
    assert batch_xyz.ndim == 3 and batch_xyz.shape[2] == 3, "Input must be of shape (bs, l, 3)"
    bs = batch_xyz.shape[0]

    # Determine subplot grid size
    rows = int(math.ceil(math.sqrt(bs)))
    cols = int(math.ceil(bs / rows))

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scatter3d'}]*cols for _ in range(rows)],
        subplot_titles=[f"Sample {i}" for i in range(bs)],
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    for idx in range(bs):
        row = idx // cols + 1
        col = idx % cols + 1

        x, y, z = batch_xyz[idx, :, 0], batch_xyz[idx, :, 1], batch_xyz[idx, :, 2]
        scatter = go.Scatter3d(
            x=x.cpu().numpy(), y=y.cpu().numpy(), z=z.cpu().numpy(),
            mode='markers',
            marker=dict(
                size=3,
                color=z.cpu().numpy(),  # color based on z
                colorscale='Viridis',
                opacity=0.8
            )
        )

        fig.add_trace(scatter, row=row, col=col)

    fig.update_layout(
        height=300*rows, width=300*cols,
        title_text=title,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.show()


def compute_lddt(ground_truth_atoms, predicted_atoms, cutoff=30.0, thresholds=[1.0, 2.0, 4.0, 8.0]):
    """
    Computes the lDDT score between ground truth and predicted atoms.
    
    Parameters:
        ground_truth_atoms (np.array): Nx3 array of ground truth atom coordinates.
        predicted_atoms (np.array): Nx3 array of predicted atom coordinates.
        cutoff (float): Distance cutoff in Ångstroms to consider neighbors. Default is 30 Å.
        thresholds (list): List of thresholds in Ångstroms for the lDDT computation. Default is [0.5, 1.0, 2.0, 4.0].
    
    Returns:
        float: The lDDT score.
    """
    # Number of atoms
    num_atoms = ground_truth_atoms.shape[0]
    
    # Initialize array to store lDDT fractions for each threshold
    fractions = np.zeros(len(thresholds))
    
    for i in range(num_atoms):
        # Get the distances from atom i to all other atoms for both ground truth and predicted atoms
        gt_distances = np.linalg.norm(ground_truth_atoms[i] - ground_truth_atoms, axis=1)
        pred_distances = np.linalg.norm(predicted_atoms[i] - predicted_atoms, axis=1)
        
        # print(gt_distances)
        # print(pred_distances)
        # exit()
        # Apply the cutoff to consider only distances within the cutoff range
        mask = (gt_distances > 0) & (gt_distances < cutoff)
        
        # Calculate the absolute difference between ground truth and predicted distances
        distance_diff = np.abs(gt_distances[mask] - pred_distances[mask])

        # Filter out any NaN values from the distance difference calculation
        valid_mask = ~np.isnan(distance_diff)
        distance_diff = distance_diff[valid_mask]

        # Compute the fractions for each threshold
        for j, threshold in enumerate(thresholds):
            if len(distance_diff)>0:
                fractions[j] += np.mean(distance_diff < threshold)
    # print(fractions)
    # print(num_atoms)

    # Average the fractions over the number of atoms
    fractions /= num_atoms
    
    # The final lDDT score is the average of these fractions
    lddt_score = np.mean(fractions)
    
    return lddt_score

import os.path as path

class CSVLogger:
    def __init__(self,columns,file):
        self.columns=columns
        self.file=file
        if not self.check_header():
            self._write_header()


    def check_header(self):
        if path.exists(self.file):
            header=True
        else:
            header=False
        return header


    def _write_header(self):
        with open(self.file,"a") as f:
            string=""
            for attrib in self.columns:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self

    def log(self,row):
        if len(row)!=len(self.columns):
            raise Exception("Mismatch between row vector and number of columns in logger")
        with open(self.file,"a") as f:
            string=""
            for attrib in row:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)






def random_rotation_point_cloud_torch(point_cloud):
    """
    Apply a random 3D rotation to a Nx3 point cloud (PyTorch version).
    
    Args:
        point_cloud (torch.Tensor): Nx3 tensor of XYZ points.

    Returns:
        torch.Tensor: Rotated Nx3 point cloud.
    """
    # Generate a random 3x3 matrix
    A = torch.randn(3, 3).to(point_cloud.device)
    # QR decomposition to get an orthonormal matrix Q
    Q, R = torch.linalg.qr(A)
    # Ensure it's a proper rotation matrix (det = +1)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    rotated = point_cloud @ Q.T
    #add random rotation as well
    

    return rotated

def random_rotation_point_cloud_torch_batch(point_clouds):
    """
    Apply a random 3D rotation to a batch of point clouds (PyTorch version).
    
    Args:
        point_clouds (torch.Tensor): BxNx3 tensor of XYZ points.

    Returns:
        torch.Tensor: Rotated BxNx3 point clouds.
    """
    B, N, _ = point_clouds.shape
    device = point_clouds.device

    # Generate a batch of random orthonormal rotation matrices
    A = torch.randn(B, 3, 3, device=device)
    Q, R = torch.linalg.qr(A)

    # Ensure det(Q) = +1 for proper rotation
    det = torch.det(Q)
    Q[det < 0, :, 0] *= -1

    # Apply batched matrix multiplication
    rotated = torch.matmul(point_clouds, Q.transpose(1, 2))  # (B, N, 3) x (B, 3, 3)^T -> (B, N, 3)

    return rotated


def calculate_distance_matrix(X,Y,epsilon=1e-4):
    return (torch.square(X[:,None]-Y[None,:])+epsilon).sum(-1).sqrt()


def dRMSD(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    if d_clamp is not None:
        rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).clip(0,d_clamp**2)
    else:
        rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon

    return rmsd.sqrt().mean()/Z


def dMSE(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    if d_clamp is not None:
        mse=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).clip(0,d_clamp**2)
    else:
        mse=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon

    return mse.mean()


def local_dRMSD(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=30):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=(~torch.isnan(gt_dm))*(gt_dm<d_clamp)
    mask[torch.eye(mask.shape[0]).bool()]=False



    rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon
    # rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).sqrt()/Z
    #rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])/Z
    #return rmsd.sqrt().mean()/Z
    return rmsd.mean()

def dRMAE(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])

    return rmsd.mean()/Z

import torch

def align_svd_mae(input, target, Z=10):
    """
    Aligns the input (Nx3) to target (Nx3) using SVD-based Procrustes alignment
    and computes RMSD loss.
    
    Args:
        input (torch.Tensor): Nx3 tensor representing the input points.
        target (torch.Tensor): Nx3 tensor representing the target points.
    
    Returns:
        aligned_input (torch.Tensor): Nx3 aligned input.
        rmsd_loss (torch.Tensor): RMSD loss.
    """
    assert input.shape == target.shape, "Input and target must have the same shape"

    #mask 
    mask=~torch.isnan(target.sum(-1))

    input=input[mask]
    target=target[mask]
    
    # Compute centroids
    centroid_input = input.mean(dim=0, keepdim=True)
    centroid_target = target.mean(dim=0, keepdim=True)

    # Center the points
    input_centered = input - centroid_input.detach()
    target_centered = target - centroid_target

    # Compute covariance matrix
    cov_matrix = input_centered.T @ target_centered

    # SVD to find optimal rotation
    U, S, Vt = torch.svd(cov_matrix)

    # Compute rotation matrix
    R = Vt @ U.T

    # Ensure a proper rotation (det(R) = 1, no reflection)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt @ U.T

    # Rotate input
    aligned_input = (input_centered @ R.T.detach()) + centroid_target.detach()

    # # Compute RMSD loss
    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())

    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())
    
    # return aligned_input, rmsd_loss
    return torch.abs(aligned_input-target).mean()/Z


def align_svd_rmsd(input, target):
    """
    Aligns the input (Nx3) to target (Nx3) using SVD-based Procrustes alignment
    and computes RMSD loss.
    
    Args:
        input (torch.Tensor): Nx3 tensor representing the input points.
        target (torch.Tensor): Nx3 tensor representing the target points.
    
    Returns:
        aligned_input (torch.Tensor): Nx3 aligned input.
        rmsd_loss (torch.Tensor): RMSD loss.
    """
    assert input.shape == target.shape, "Input and target must have the same shape"

    #mask 
    mask=~torch.isnan(target.sum(-1))

    input=input[mask]
    target=target[mask]
    
    # Compute centroids
    centroid_input = input.mean(dim=0, keepdim=True)
    centroid_target = target.mean(dim=0, keepdim=True)

    # Center the points
    input_centered = input - centroid_input.detach()
    target_centered = target - centroid_target

    # Compute covariance matrix
    cov_matrix = input_centered.T @ target_centered

    # SVD to find optimal rotation
    U, S, Vt = torch.svd(cov_matrix)

    # Compute rotation matrix
    R = Vt @ U.T

    # Ensure a proper rotation (det(R) = 1, no reflection)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt @ U.T

    # Rotate input
    aligned_input = (input_centered @ R.T.detach()) + centroid_target.detach()

    # # Compute RMSD loss
    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())

    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())
    
    # return aligned_input, rmsd_loss
    return torch.square(aligned_input-target).mean().sqrt()