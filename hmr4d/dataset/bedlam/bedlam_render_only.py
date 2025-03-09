import numpy as np
import torch
from pathlib import Path
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer_utils import simple_render_mesh_background
from hmr4d.utils.video_io_utils import read_video_np, save_video
from concurrent.futures import ThreadPoolExecutor
import cv2

# Load BEDLAM data
data_path = "inputs/bedlam_30fps/training_labels_30fps/20221010_3_1000_batch01hand.npz"
data = np.load(data_path)

# Base directory where images are stored
image_base_dir = "inputs/data/b0_all/20221010_3_1000_batch01hand/png/"

# Construct full image paths using NumPy string operations
image_filenames = np.char.add(image_base_dir, data["imgnames"])  # Efficient path concatenation

# Function to load a single image
def load_image(img_path):
    img = cv2.imread(img_path)  # Load in BGR format
    if img is None:
        print(f"Warning: Image not found")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return torch.tensor(img, dtype=torch.uint8).permute(2, 0, 1)  # (H, W, C) → (C, H, W)

# Load images in parallel using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    images = list(executor.map(load_image, image_filenames))

# Remove None values (in case some images are missing)
images = [img for img in images if img is not None]

# Convert to tensor batch if images exist
images = torch.stack(images) if images else None
import ipdb;ipdb.set_trace()

# Extract required parameters
betas = torch.tensor(data["betas"][0][:10], dtype=torch.float32)  # Shape coefficients (only first 10)
body_pose = torch.tensor(data["poses_cam"][0][3:66], dtype=torch.float32)  # Body pose (excluding global rotation)
global_orient = torch.tensor(data["poses_cam"][0][:3], dtype=torch.float32)  # Global orientation
transl = torch.tensor(data["trans_cam"][0], dtype=torch.float32)  # Translation

# Create SMPL-X model
smplx_model = make_smplx("supermotion")

# Generate mesh
smplx_out = smplx_model(
    betas=betas.unsqueeze(0), 
    body_pose=body_pose.unsqueeze(0), 
    global_orient=global_orient.unsqueeze(0), 
    transl=transl.unsqueeze(0)
)

# Prepare rendering input
render_dict = {
    "K": torch.tensor(data["cam_int"][0]).unsqueeze(0),  # Camera intrinsics
    "faces": smplx_model.faces,  # SMPL-X mesh faces
    "verts": smplx_out.vertices,  # Generated vertices
    "background": None,  # Set this to an image if available
}

# Render and save the result
rendered_img = simple_render_mesh_background(render_dict)
save_video(rendered_img, "outputs/bedlam_render.mp4", crf=23)

print("Rendering complete! Saved as 'bedlam_render.mp4'.")


'''
import numpy as np
import torch
from pathlib import Path
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer_utils import simple_render_mesh_background
from hmr4d.utils.video_io_utils import read_video_np, save_video

# Load BEDLAM data
data_path = "20221010_3_1000_batch01hand.npz"
data = np.load(data_path)

# SMPL params in cam
#body_pose = data["pose"][:, 3:]  # (F, 63)
body_pose = data["poses_cam"][:, 3:66]
#betas = data["beta"].repeat(length, 1)  # (F, 10)
betas = data["betas"][:, :10].repeat(length, 1)
#global_orient = data["global_orient_incam"]  # (F, 3)
global_orient = data["poses_cam"][:, :3]
#transl = data["trans_incam"] + data["cam_ext"][:, :3, 3]  # (F, 3), bedlam convention
transl = data["trans_cam"] + data["cam_ext"][:3, 3]
smpl_params_c = {"body_pose": body_pose, "betas": betas, "transl": transl, "global_orient": global_orient}

# Create SMPL-X model
smplx_model = make_smplx("supermotion")

# Generate mesh
smplx_out = smplx(**smpl_params_c)

# Prepare rendering input
render_dict = {
    "K": data["cam_int"][:1],  # only support batch-size 1
    "faces": smplx.faces,
    "verts": smplx_out.vertices,
    "background": images,
}
img_overlay = simple_render_mesh_background(render_dict)
save_video(img_overlay, "tmp.mp4", crf=23)


# Render and save the result
rendered_img = simple_render_mesh_background(render_dict)
save_video(rendered_img, "bedlam_render.mp4", crf=23)

print("Rendering complete! Saved as 'bedlam_render.mp4'.")

'''


