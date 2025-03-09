import numpy as np
import torch
from pathlib import Path
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer_utils import simple_render_mesh_background
from hmr4d.utils.video_io_utils import read_video_np, save_video
from concurrent.futures import ThreadPoolExecutor
import cv2

# Load BEDLAM data
data_path = "inputs/bedlam_30fps/training_labels_30fps/first_entry.npz"
data = np.load(data_path)

# Extract required parameters
betas = torch.tensor(data["betas"][:10], dtype=torch.float32)  # Shape coefficients (only first 10)
body_pose = torch.tensor(data["poses_cam"][3:66], dtype=torch.float32)  # Body pose (excluding global rotation)
global_orient = torch.tensor(data["poses_cam"][:3], dtype=torch.float32)  # Global orientation
transl = torch.tensor(data["trans_cam"], dtype=torch.float32)  # Translation

# Create SMPL-X model
smplx_model = make_smplx("supermotion")

# Generate mesh
smplx_out = smplx_model(
    betas=betas.unsqueeze(0), 
    body_pose=body_pose.unsqueeze(0), 
    global_orient=global_orient.unsqueeze(0), 
    transl=transl.unsqueeze(0)
)

# Load and process the background image
background_img_path = "inputs/data/b0_all/20221010_3_1000_batch01hand/png/seq_000000/seq_000000_0000.png"  # Provide the path to your image
image = cv2.imread(background_img_path)  # Load image with OpenCV
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (OpenCV loads as BGR)
image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor (C, H, W, 1 batch)

# Prepare rendering input
render_dict = {
    "K": torch.tensor(data["cam_int"][0]).unsqueeze(0),  # Camera intrinsics
    "faces": smplx_model.faces,  # SMPL-X mesh faces
    "verts": smplx_out.vertices,  # Generated vertices
    "background": image,  # Set this to an image if available
}

# Render and save the result
rendered_img = simple_render_mesh_background(render_dict)
save_video(rendered_img, "outputs/bedlam_render.png", crf=23)

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


