import torch
import numpy as np

class MockMeshGenerator:
    def __init__(self, device='cpu'):
        self.device = device

    def generate_plane(self, res=64, size=0.5):
        x = torch.linspace(-size/2, size/2, res)
        y = torch.linspace(-size/2, size/2, res)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        z = torch.zeros_like(grid_x)
        verts = torch.stack([grid_x, grid_y, z], dim=-1).reshape(-1, 3).to(self.device)
        faces = []
        for i in range(res - 1):
            for j in range(res - 1):
                idx = i * res + j
                faces.append([idx, idx + res, idx + 1])
                faces.append([idx + 1, idx + res, idx + res + 1])
        faces = torch.tensor(faces, dtype=torch.long).to(self.device)
        return verts, faces

    def apply_perturbation(self, verts, noise_level=0.005, warp_strength=0.0):
        v_out = verts.clone()
        if noise_level > 0:
            noise = torch.randn_like(v_out) * noise_level
            v_out += noise
        return v_out

    def add_bump_detail(self, verts, center=(0,0), radius=0.1, height=0.03):
        v_out = verts.clone()
        cx, cy = center
        dists = torch.sqrt((v_out[:, 0] - cx)**2 + (v_out[:, 1] - cy)**2)
        mask = dists < radius
        if mask.any():
            falloff = 0.5 * (1 + torch.cos(3.14159 * dists[mask] / radius))
            v_out[mask, 2] += height * falloff
        return v_out
