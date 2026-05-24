import gradio as gr
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import tempfile
import imageio.v2 as imageio
import cv2

# Import from your existing modules
from model import (
    BasicVSRWrapper,
    EDVRWrapper,
    RSDNWrapper,
    OVSRWrapper
)

# 1. Models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

models = {
        'BasicVSR': BasicVSRWrapper(checkpoint_path='./checkpoints/vsrnet_checkpoints/basicvsr_reds4_20120409-0e599677.pth', device=device),
        'EDVR': EDVRWrapper(checkpoint_path='./checkpoints/edvr_checkpoints/edvrl_c128b40_8x8_lr2e-4_600k_reds4_20220104-4509865f.pth', device=device),
        'RSDN': RSDNWrapper(checkpoint_path='./checkpoints/rsdn_checkpoints/RSDN.pth', device=device),
        'OVSR': OVSRWrapper(checkpoint_path='./checkpoints/ovsr_checkpoints/0721.pth', device=device)
    }
# Set eval mode safely
for model in models.values():

    if hasattr(model, 'eval'):
        model.eval()

    elif hasattr(model, 'model'):
        model.model.eval()


# 2. Degradation 

def apply_degradation(img, method, level):

    w, h = img.size

    scale_factor = int(level)

    align_factor = 8

    new_w = w - (w % align_factor)
    new_h = h - (h % align_factor)

    img = img.crop((0, 0, new_w, new_h))

    if method == "Bicubic Downsample":
        lr_img = img.resize((new_w // scale_factor, new_h // scale_factor), Image.BICUBIC)
        display_img = lr_img.resize((new_w, new_h), Image.NEAREST)
        return lr_img, display_img

    elif method == "Bicubic Interpolated":
        lr_img = img.resize((new_w // scale_factor, new_h // scale_factor), Image.BICUBIC)
        display_img = lr_img.resize((new_w, new_h), Image.BICUBIC)
        return lr_img, display_img

    return img, img

# 3. Save Video 

def save_video_imageio(path, frames, fps):

    if len(frames) == 0:
        raise ValueError("No frames to save.")

    try:
        fps = float(fps)
    except:
        fps = 30.0

    if (np.isnan(fps) or np.isinf(fps) or fps <= 0 or fps > 120):
        fps = 30.0

    processed_frames = []

    for frame in frames:
        frame = frame.astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        processed_frames.append(frame)

    imageio.mimwrite(path, processed_frames, fps=fps, codec='libx264', format='FFMPEG', ffmpeg_params=['-pix_fmt', 'yuv420p'])

# 4. Image inference

def process_image(input_img, deg_method, deg_level, model_name):

    try:

        if input_img is None:
            return None, None

        # Original resolution
        original_w, original_h = input_img.size

        # Apply degradation
        lr_img, display_deg_img = apply_degradation(
            input_img,
            deg_method,
            deg_level
        )

        # To tensor
        lr_tensor = TF.to_tensor(lr_img)

        # Create fake temporal sequence
        seq_tensor = (
            lr_tensor
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 3, 1, 1, 1)
            .to(device)
        )

        # Select model
        model = models[model_name]

        # Inference
        with torch.inference_mode():

            output = model(seq_tensor)

        # Handle output shape
        if output.dim() == 5:

            out_tensor = output[0, output.shape[1] // 2]

        elif output.dim() == 4:

            out_tensor = output[0]

        else:

            raise ValueError(
                f"Unexpected output shape: {output.shape}"
            )

        # Clamp
        out_tensor = output[0].cpu().clamp(0, 1)

        # Tensor -> PIL
        out_img = TF.to_pil_image(out_tensor)

        # Resize back to original resolution
        out_img = out_img.resize(
            (original_w, original_h),
            Image.BICUBIC
        )

        # Cleanup
        del seq_tensor
        del output

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return display_deg_img, out_img

    except Exception as e:

        print("IMAGE ERROR:", e)

        return None, None


# 5. Video inference

def process_video(input_video_path, deg_method, deg_level, model_name):

    try:

        if input_video_path is None:
            return None, None

        # Gradio video returns dict sometimes
        if isinstance(input_video_path, dict):

            input_video_path = input_video_path["path"]

        # Read video
        cap = cv2.VideoCapture(input_video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        try:

            fps = float(fps)

        except:

            fps = 30.0

        # Fix invalid FPS
        if (
            np.isnan(fps)
            or np.isinf(fps)
            or fps <= 0
            or fps > 120
        ):

            fps = 30.0

        frames = []

        MAX_FRAMES = 150

        count = 0

        # Read frames
        while cap.isOpened() and count < MAX_FRAMES:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(Image.fromarray(frame))

            count += 1

        cap.release()

        if len(frames) == 0:

            raise ValueError(
                "Could not read video frames."
            )

        # Original resolution
        original_w, original_h = frames[0].size

        # Degradation
        lr_frames = []

        display_deg_frames = []

        for img in frames:

            lr_img, display_deg_img = apply_degradation(
                img,
                deg_method,
                deg_level
            )

            lr_frames.append(
                TF.to_tensor(lr_img)
            )

            display_deg_frames.append(
                np.array(display_deg_img)
            )

        # Create sequence tensor
        seq_tensor = (
            torch.stack(lr_frames, dim=0)
            .unsqueeze(0)
            .to(device)
        )

        # Select model
        model = models[model_name]

        # Inference
        with torch.inference_mode():

            output = model(seq_tensor)

        # Handle output shape
        if output.dim() == 5:

            out_seq = output[0]

        elif output.dim() == 4:

            out_seq = output

        else:

            raise ValueError(
                f"Unexpected output shape: {output.shape}"
            )

        # Clamp
        out_seq = out_seq.cpu().clamp(0, 1)

        out_frames = []

        # Convert output frames
        for i in range(out_seq.size(0)):

            out_img = TF.to_pil_image(
                out_seq[i]
            )

            # Resize back to original resolution
            out_img = out_img.resize(
                (original_w, original_h),
                Image.BICUBIC
            )

            out_frames.append(
                np.array(out_img)
            )

        # Save videos
        deg_temp_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.mp4'
        ).name

        out_temp_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.mp4'
        ).name

        save_video_imageio(
            deg_temp_path,
            display_deg_frames,
            fps
        )

        save_video_imageio(
            out_temp_path,
            out_frames,
            fps
        )

        # Cleanup
        del seq_tensor
        del output

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return deg_temp_path, out_temp_path

    except Exception as e:

        print("VIDEO ERROR:", e)

        return None, None
# 6. Gradio UI

with gr.Blocks(title="Video Super Resolution Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "<h1 style='text-align: center;'>"
        "Video & Image Super Resolution Live Demo"
        "</h1>"
    )

    with gr.Row():
        with gr.Column():
            deg_method = gr.Radio(choices=["Bicubic Downsample", "Bicubic Interpolated"], value="Bicubic Downsample", label="Degradation Method")
            deg_level = gr.Slider(minimum=2, maximum=10, value=4, step=1, label="Scale Factor")


        with gr.Column():
            model_sel = gr.Dropdown(choices=["BasicVSR", "EDVR", "RSDN", "OVSR"], value="BasicVSR", label="Select VSR Model")

    with gr.Tabs():


        with gr.TabItem("Video super resolution (.mp4)"):

            gr.Markdown(
                "Upload a short video "
                "(up to 30 frames processed)."
            )

            with gr.Row():

                vid_in = gr.Video(label="1. Original Video")
                vid_deg = gr.Video(label="2. Degraded Video", interactive=False)
                vid_out = gr.Video(label="3. Super Resolved Video", interactive=False)

            process_vid_btn = gr.Button("Run Super Resolution (VIDEO)", variant="primary")

        
        with gr.TabItem("Image super resolution"):

            gr.Markdown(
                "Upload a static image. "
                "The system duplicates it "
                "into a short sequence."
            )

            with gr.Row():

                img_in = gr.Image(label="1. Original Image", type="pil")
                img_deg = gr.Image(label="2. Degraded Image", type="pil", interactive=False)
                img_out = gr.Image(label="3. Super Resolved Image", type="pil", interactive=False)

            process_img_btn = gr.Button("Run Super Resolution (IMAGE)", variant="primary")


    process_vid_btn.click(fn=process_video, inputs=[vid_in, deg_method, deg_level, model_sel], \
                            outputs=[vid_deg, vid_out])

    process_img_btn.click(fn=process_image, inputs=[img_in, deg_method, deg_level, model_sel], \
                            outputs=[img_deg, img_out])

# 7. Launch


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)