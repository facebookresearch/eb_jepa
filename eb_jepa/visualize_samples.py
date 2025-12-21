import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_images(
    tensor,
    nrow=4,
    titles=None,
    labels=None,
    save_path=None,
    dpi=150,
    close_fig=True,
    first_channel_only=True,
    clamp=True,
):
    """
    Display and optionally save a grid of images from a PyTorch tensor
    Args:
        tensor: Input tensor of shape (B, C, H, W) or (B, T, C, H, W)
        nrow: Number of images per row in the grid
        titles: List of titles for each image
        labels: List of labels for each image
        save_path: Path to save figure (None to disable saving)
        dpi: Resolution for saved figure
        close_fig: Whether to close figure after saving/displaying
    """
    # Convert to CPU and detach from computation graph
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()

    # Handle 5D tensors (batch, time, channel, height, width)
    if tensor.ndim == 5:
        tensor = tensor[:, 0]  # Use first frame for visualization

    # Add channel dimension handling
    if tensor.ndim == 4 and first_channel_only:
        tensor = tensor[:, 0:1]  # Keep only first channel

    # Convert to numpy and denormalize (assuming [0,1] range)
    if clamp:
        tensor = tensor.clamp(0, 1).numpy()

    # Create plot
    batch_size = tensor.shape[0]
    ncol = min(nrow, batch_size)
    nrow = (batch_size + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2), dpi=dpi)
    if nrow == 1 and ncol == 1:
        axes = [[axes]]  # Ensure 2D array for single image

    for i, ax in enumerate(axes.flat):
        if i >= batch_size:
            ax.axis("off")
            continue
        img = tensor[i].squeeze()
        if img.ndim == 3 and img.shape[0] < 3:
            h, w = img.shape[1], img.shape[2]
            rgb_img = np.zeros((h, w, 3))
            for c in range(min(img.shape[0], 3)):
                rgb_img[..., c] = img[c]
            img = rgb_img.astype(np.uint8)
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i], fontsize=10)
        if labels:
            ax.text(
                0.5,
                -0.15,
                labels[i],
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )

    plt.tight_layout()

    # Save figure if path specified
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)

    # Display figure if not closed
    if not close_fig or not save_path:
        plt.show()

    # Clean up
    if close_fig:
        plt.close(fig)


def save_gif(tensor, save_path, fps=10, show_frame_numbers=False):
    """
    tensor of shape (T, C, H, W) uint8
    """

    # Save each frame as an image
    images = []
    total_frames = tensor.shape[0]
    for i in range(total_frames):
        img = tensor[i].numpy().astype(np.uint8)
        if img.ndim == 3 and img.shape[0] < 3:  # Handle fewer than 3 channels
            h, w = img.shape[1], img.shape[2]
            rgb_img = np.zeros((h, w, 3))
            for c in range(min(img.shape[0], 3)):
                rgb_img[..., c] = img[c]
            img = rgb_img.astype(np.uint8)
        else:
            img = img.squeeze()
        if show_frame_numbers:
            # Calculate scale factors based on image dimensions
            h, w = img.shape[0], img.shape[1]
            scale_factor = min(h, w) / 1000  # Base scale on 500px reference
            font_scale = max(0.2, scale_factor * 0.5)
            thickness = max(1, int(scale_factor))
            margin = int(h * 0.02)  # 3% of height

            # Add frame number text
            text = f"Frame {i+1}/{total_frames}"

            # Get text size to position it at top right with margin
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            text_x = w - text_width - margin
            text_y = text_height + margin

            cv2.putText(
                img,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        images.append(img)

    # Save as GIF
    imageio.mimsave(save_path, images, fps=fps, loop=0)


def to3channels(input_array, channel_dim=2):
    """
    Convert a tensor or numpy array with fewer than 3 channels to 3 channels by adding zeros.
    Preserves the input dtype and type (numpy array or PyTorch tensor).

    Args:
        input_array: Input tensor or numpy array of shape ... C ... at idx channel_dim
                    where C can be < 3.
        channel_dim: Dimension index where channels are located (default is 2 for (H, W, C) format)

    Returns:
        A tensor or numpy array of the same type and dtype as input_array, but with 3 channels.
    """
    is_torch_tensor = isinstance(input_array, torch.Tensor)
    ndim = input_array.ndim
    if ndim < channel_dim + 1:
        raise ValueError(f"Input must have at least {channel_dim + 1} dimensions.")
    shape = list(input_array.shape)
    if shape[channel_dim] >= 3:
        return input_array
    new_shape = shape.copy()
    new_shape[channel_dim] = 3

    if is_torch_tensor:
        new_array = torch.zeros(
            new_shape, dtype=input_array.dtype, device=input_array.device
        )
    else:
        new_array = np.zeros(new_shape, dtype=input_array.dtype)

    # Create a dynamic slice for the channel dimension
    indices = [slice(None)] * ndim
    indices[channel_dim] = slice(0, shape[channel_dim])
    new_array[tuple(indices)] = input_array

    return new_array


def save_gif_HWC(frames_list, save_path, fps=10):
    """
    Save a list of image tensors as a GIF.

    Args:
        frames_list: List of tensors, each with shape (H, W, C) where C can be < 3
        save_path: Path to save the GIF
        fps: Frames per second
    """
    images = []

    for frame in frames_list:
        # Convert to numpy if it's a tensor
        if isinstance(frame, torch.Tensor):
            img = frame.detach().cpu().numpy()
        else:
            img = np.array(frame)

        # Ensure uint8 type
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Handle case where C < 3
        if img.ndim == 3 and img.shape[2] < 3:
            h, w = img.shape[0], img.shape[1]
            rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(min(img.shape[2], 3)):
                rgb_img[..., c] = img[..., c]
            images.append(rgb_img)
        # Handle grayscale case (H, W)
        elif img.ndim == 2:
            h, w = img.shape
            rgb_img = np.stack([img, img, img], axis=2)
            images.append(rgb_img)
        else:
            images.append(img)

    # Save as GIF
    imageio.mimsave(save_path, images, fps=fps, loop=0)


