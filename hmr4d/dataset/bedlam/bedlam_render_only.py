
import numpy as np
import torch
import cv2
from pathlib import Path
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer_utils import simple_render_mesh_background
from hmr4d.utils.video_io_utils import save_video  # Import the save_video function

def data_loader(data_path, sequence_name):
    """Load data for a specific sequence from the .npz file."""
    data = np.load(data_path)

    # Extract sequence-specific parameters from the data
    sequence_imgnames = [imgname for imgname in data["imgnames"] if sequence_name in imgname]
    
    # Extract parameters for SMPL-X
    betas = torch.tensor(data["betas"][:,:10], dtype=torch.float32)  # Shape coefficients (first 10)
    body_pose = torch.tensor(data["poses_cam"][:,3:66], dtype=torch.float32)  # Body pose (excluding global rotation)
    global_orient = torch.tensor(data["poses_cam"][:,:3], dtype=torch.float32)  # Global orientation
    transl = torch.tensor(data["trans_cam"] + data["cam_ext"][:, :3, 3], dtype=torch.float32)  # Translation
    smpl_params_c = {"body_pose": body_pose, "betas": betas, "transl": transl, "global_orient": global_orient}
    # Create SMPL-X model
    smplx_model = make_smplx("supermotion")

    # Camera intrinsics (assuming these are shared across sequences)
    K = torch.tensor(data["cam_int"][:1])

    return smplx_model, smpl_params_c, sequence_imgnames, K

def load_background_image(image_path):
    """Load and process the background image."""
    image = cv2.imread(image_path)  # Load image with OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (OpenCV loads as BGR)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor (C, H, W, 1 batch)
    return image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to NumPy array (H, W, C)

def renderer(smplx_model, smpl_params_c, K, img_path):
    """Render a single frame."""
    # Generate mesh for the current frame
    smplx_out = smplx_model(**smpl_params_c)
    
    # Load the corresponding background image
    background = load_background_image(img_path)

    # Prepare rendering input
    render_dict = {
        "K": K,  # Camera intrinsics
        "faces": smplx_model.faces,  # SMPL-X mesh faces
        "verts": smplx_out.vertices,  # Generated vertices
        "background": background,  # Set this to an image if available
    }
    
    # Render the image
    rendered_img = simple_render_mesh_background(render_dict)
    rendered_img = rendered_img.squeeze(0)  # Remove the batch dimension

    return rendered_img

def create_video(data_path, output_dir, fps=30, crf=17):
    """Load data, render frames, and create a video for each sequence."""
    
    # Read the sequence names from the .npz file (or can be inferred from image paths)
    data = np.load(data_path)
    sequence_names = set([imgname.split("/")[0] for imgname in data["imgnames"]])

    # Process each sequence individually
    for sequence_name in sequence_names:
        print(f"Processing sequence: {sequence_name}")
        
        # Load sequence-specific data
        smplx_model, smpl_params_c, sequence_imgnames, K = data_loader(data_path, sequence_name)
        frames = []

        for j, img_path in enumerate(sequence_imgnames):
            # Extract only the parameters for the current frame
            smpl_params_single = {
                    "body_pose": smpl_params_c["body_pose"][j].unsqueeze(0),
                    "betas": smpl_params_c["betas"][j].unsqueeze(0),
                    "transl": smpl_params_c["transl"][j].unsqueeze(0),
                    "global_orient": smpl_params_c["global_orient"][j].unsqueeze(0),
            }

            img_path = Path("inputs/data/b0_all/20221010_3_1000_batch01hand/png") / img_path
            rendered_img = renderer(smplx_model, smpl_params_single, K, img_path)
            #rendered_img = np.clip(rendered_img, 0, 255).astype(np.uint8)
            frames.append(rendered_img)

        # Define the output path for the video of the current sequence
        output_video_path = Path(output_dir) / f"{sequence_name}_rendered.mp4"
        
        # Save the video using the save_video function
        save_video(frames, output_video_path, fps=fps, crf=crf)
        print(f"Video for sequence {sequence_name} saved to {output_video_path}")

# Example usage
data_path = "inputs/bedlam_30fps/training_labels_30fps/seq_000000_data.npz"
output_dir = "outputs/bedlam_render_videos"
create_video(data_path, output_dir, fps=30, crf=17)

'''
#render only one image
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
transl = torch.tensor(data["trans_cam"]+data["cam_ext"][:3, 3], dtype=torch.float32)  # Translation

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
image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) format and then to NumPy array

# Prepare rendering input
render_dict = {
    "K": torch.tensor(data["cam_int"]).unsqueeze(0),  # Camera intrinsics
    "faces": smplx_model.faces,  # SMPL-X mesh faces
    "verts": smplx_out.vertices,  # Generated vertices
    "background": image_np,  # Set this to an image if available
}

# Render and save the result
rendered_img = simple_render_mesh_background(render_dict)
rendered_img = rendered_img.squeeze(0)  # This will remove the first dimension (1)
print(rendered_img.shape)
cv2.imwrite("outputs/bedlam_render.png", rendered_img)
#save_video(rendered_img, "outputs/bedlam_render.png", crf=23)

print("Rendering complete! Saved as 'bedlam_render.mp4'.")
'''