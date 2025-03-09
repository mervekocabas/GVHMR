from hmr4d.utils.vis.renderer import Renderer
from tqdm import tqdm
import numpy as np
import torch

def simple_render_mesh(render_dict):
    """Render an camera-space mesh, blank background"""
    width, height, focal_length = render_dict["whf"]
    faces = render_dict["faces"]
    verts = render_dict["verts"]

    renderer = Renderer(width, height, focal_length, device="cuda", faces=faces)
    outputs = []
    for i in tqdm(range(len(verts)), desc=f"Rendering"):
        img = renderer.render_mesh(verts[i].cuda(), colors=[0.8, 0.8, 0.8])
        outputs.append(img)
    outputs = np.stack(outputs, axis=0)
    return outputs


def simple_render_mesh_background(render_dict, VI=50, colors=[0.8, 0.8, 0.8]):
    """Render an camera-space mesh, blank background"""
    K = render_dict["K"]
    faces = render_dict["faces"]
    verts = render_dict["verts"]
    background = render_dict["background"]
    N_frames = len(verts)
    if len(background.shape) == 3:
        background = [background] * N_frames
    height, width = background[0].shape[:2]

    renderer = Renderer(width, height, device="cuda", faces=faces, K=K)
    outputs = []
    for i in tqdm(range(len(verts)), desc=f"Rendering"):
        img = renderer.render_mesh(verts[i].cuda(), colors=colors, background=background[i], VI=VI)
        outputs.append(img)
    outputs = np.stack(outputs, axis=0)
    return outputs

def simple_render_mesh_solid_background(render_dict, VI=50, colors=[0.8, 0.8, 0.8], bg_color=[0, 0, 0]):
    """Render a camera-space mesh with an optional blank background."""
    K = render_dict["K"]
    faces = render_dict["faces"]
    verts = render_dict["verts"]
    background = render_dict["background"]

    N_frames = len(verts)

    # If no background is provided, create an empty background
    if background is None or len(background) == 0:
        height, width = 512, 512  # Default size, adjust as needed
        background = [torch.full((3, height, width), bg_color[i], dtype=torch.uint8) for i in range(3)]
        background = torch.stack(background, dim=0).permute(1, 2, 0)  # Convert to (H, W, C)
        background = [background] * N_frames  # Duplicate for all frames
    elif len(background.shape) == 3:
        background = [background] * N_frames

    height, width = background[0].shape[:2]

    renderer = Renderer(width, height, device="cuda", faces=faces, K=K)
    outputs = []
    for i in tqdm(range(N_frames), desc="Rendering"):
        img = renderer.render_mesh(verts[i].cuda(), colors=colors, background=background[i], VI=VI)
        outputs.append(img)

    outputs = np.stack(outputs, axis=0)
    return outputs
