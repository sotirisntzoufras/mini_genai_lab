# ============================================================
# MINI GENAI LAB v1
# Local educational application for architecture students
# Dataset + Autoencoder + VAE + Generation + Interpolation + Compare
# ============================================================

import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import json
import hashlib
import math
import random
import shutil
import subprocess
import sys
import time
import platform
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# PROJECT PATHS
# All paths are anchored to the folder containing this app.py.
# This prevents outputs from being written to an unexpected folder.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
RAW_DATASET_DIR = DATASETS_DIR / "raw"
SOURCE_GEOMETRY_DIR = DATASETS_DIR / "source_geometry"
TEST_INPUTS_DIR = DATASETS_DIR / "test_inputs"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATASET_OUTPUT_DIR = OUTPUTS_DIR / "dataset"
AE_TRAINING_DIR = OUTPUTS_DIR / "autoencoder" / "training"
AE_RECON_DIR = OUTPUTS_DIR / "autoencoder" / "reconstructions"
VAE_TRAINING_DIR = OUTPUTS_DIR / "vae" / "training"
VAE_RECON_DIR = OUTPUTS_DIR / "vae" / "reconstructions"
VAE_GENERATION_DIR = OUTPUTS_DIR / "vae" / "generations"
VAE_INTERPOLATION_DIR = OUTPUTS_DIR / "vae" / "interpolations"
COMPARE_DIR = OUTPUTS_DIR / "compare"

AUTOENCODER_MODEL_PATH = MODELS_DIR / "shapes_autoencoder.pt"
VAE_MODEL_PATH = MODELS_DIR / "shapes_vae.pt"
AUTOENCODER_CONFIG_PATH = MODELS_DIR / "autoencoder_config.json"
VAE_CONFIG_PATH = MODELS_DIR / "vae_config.json"
DATASET_CONFIG_PATH = DATASETS_DIR / "latest_dataset_config.json"
SOURCE_GEOMETRY_PATH = SOURCE_GEOMETRY_DIR / "latest_source_geometry.png"

IMAGE_SIZE = 64
CATEGORIES = ["circle", "square", "triangle", "rectangle", "cross"]
DEFAULT_SEED = 42
KL_WEIGHT = 1.0
CUSTOM_CLASS_NAME = "student_source"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_dirs():
    """Create only the permanent project folders."""
    for folder in [
        DATASETS_DIR,
        RAW_DATASET_DIR,
        SOURCE_GEOMETRY_DIR,
        TEST_INPUTS_DIR,
        MODELS_DIR,
        DATASET_OUTPUT_DIR,
        AE_TRAINING_DIR,
        AE_RECON_DIR,
        VAE_TRAINING_DIR,
        VAE_RECON_DIR,
        VAE_GENERATION_DIR,
        VAE_INTERPOLATION_DIR,
        COMPARE_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_device_label():
    if torch.cuda.is_available():
        try:
            return f"CUDA - {torch.cuda.get_device_name(0)}"
        except Exception:
            return "CUDA"
    return "CPU"


def runtime_metadata():
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "device_label": compute_device_label(),
        "seed": DEFAULT_SEED,
    }


def current_dataset_config():
    return load_json(DATASET_CONFIG_PATH) or {}


def current_dataset_id():
    return current_dataset_config().get("dataset_id")


def save_latest_image(image, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / f"latest_{name}.png"
    image.save(latest)
    return latest


def image_signature(image):
    """Hash the 64x64 grayscale representation actually used by the models."""
    tensor = base_transform()(image.convert("L")).squeeze(0).cpu().numpy()
    quantized = np.clip(np.rint(tensor * 255.0), 0, 255).astype(np.uint8)
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def open_pil_copy(path):
    with Image.open(path) as img:
        return img.copy()


def project_relative(path):
    """Return a clean project-relative path for messages shown in the UI."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def storage_summary():
    return (
        f"Project root: {BASE_DIR}\n"
        f"Training dataset: {RAW_DATASET_DIR}\n"
        f"Dataset config: {DATASET_CONFIG_PATH}\n"
        f"Source geometry: {SOURCE_GEOMETRY_DIR}\n"
        f"Models: {MODELS_DIR}\n"
        f"Saved outputs: {OUTPUTS_DIR}\n"
        f"Compute device: {compute_device_label()}"
    )


def open_outputs_folder():
    """Open the local outputs folder in the operating system file browser."""
    ensure_dirs()
    try:
        if os.name == "nt":
            os.startfile(str(OUTPUTS_DIR))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(OUTPUTS_DIR)])
        else:
            subprocess.Popen(["xdg-open", str(OUTPUTS_DIR)])
        return f"Opened local folder: {OUTPUTS_DIR}"
    except Exception as exc:
        return f"Could not open the folder automatically. Open it manually here: {OUTPUTS_DIR}\n{exc}"


def clear_experiment_outputs(confirm):
    """Delete only the outputs folder and recreate the v1 output structure."""
    if not confirm:
        return (
            "Nothing was deleted. Tick the confirmation box first. "
            "This action deletes saved output images, logs, and output history, but keeps datasets and trained models."
        )

    try:
        if OUTPUTS_DIR.exists():
            shutil.rmtree(OUTPUTS_DIR)
        ensure_dirs()
        return (
            "Saved experiment outputs were cleared successfully.\n"
            "The training dataset and trained model files were preserved.\n"
            f"New output structure created at: {OUTPUTS_DIR}"
        )
    except Exception as exc:
        return f"Could not clear the output folder: {exc}"


def save_pil_history(image, folder, stem):
    """Save a timestamped image and update a latest_*.png copy."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    unique_path = folder / f"{stem}_{timestamp()}.png"
    latest_path = folder / f"latest_{stem}.png"
    image.save(unique_path)
    image.save(latest_path)
    return unique_path, latest_path


def save_figure_history(fig, folder, stem, dpi=200):
    """Save a timestamped matplotlib figure and update a latest_*.png copy."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    unique_path = folder / f"{stem}_{timestamp()}.png"
    latest_path = folder / f"latest_{stem}.png"
    fig.savefig(unique_path, dpi=dpi, bbox_inches="tight")
    shutil.copy2(unique_path, latest_path)
    plt.close(fig)
    return unique_path, latest_path


def save_text_history(text, folder, stem):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    unique_path = folder / f"{stem}_{timestamp()}.txt"
    latest_path = folder / f"latest_{stem}.txt"
    unique_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    return unique_path, latest_path


def base_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


def prepare_input_image(uploaded_image):
    original = uploaded_image.convert("L")
    tensor = base_transform()(original).unsqueeze(0).to(device)
    return original, tensor


def tensor_to_pil(tensor):
    tensor = tensor.detach().squeeze().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor).convert("L")


def make_side_by_side(original_pil, reconstructed_pil, left_title, right_title):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), squeeze=False)
    axes[0, 0].imshow(original_pil, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(left_title)
    axes[0, 0].axis("off")
    axes[0, 1].imshow(reconstructed_pil, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(right_title)
    axes[0, 1].axis("off")
    fig.tight_layout()
    return fig


def make_three_panel(original_pil, ae_pil, vae_pil):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), squeeze=False)
    titles = ["Post-training Input", "Autoencoder Reconstruction", "VAE Reconstruction"]
    images = [original_pil, ae_pil, vae_pil]
    for i, (title, image) in enumerate(zip(titles, images)):
        axes[0, i].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(title)
        axes[0, i].axis("off")
    fig.tight_layout()
    return fig


# ============================================================
# DATASET GENERATION
# ============================================================

def random_shape_size():
    return random.randint(24, 44)


def random_position(width, height):
    x = random.randint(4, IMAGE_SIZE - width - 4)
    y = random.randint(4, IMAGE_SIZE - height - 4)
    return x, y


def draw_circle(draw):
    d = random_shape_size()
    x, y = random_position(d, d)
    draw.ellipse([x, y, x + d, y + d], fill=255)


def draw_square(draw):
    s = random_shape_size()
    x, y = random_position(s, s)
    draw.rectangle([x, y, x + s, y + s], fill=255)


def draw_rectangle(draw):
    w = random.randint(28, 46)
    h = random.randint(18, 32)
    x, y = random_position(w, h)
    draw.rectangle([x, y, x + w, y + h], fill=255)


def draw_triangle(draw):
    s = random_shape_size()
    x, y = random_position(s, s)
    points = [(x + s // 2, y), (x, y + s), (x + s, y + s)]
    draw.polygon(points, fill=255)


def draw_cross(draw):
    length = random.randint(26, 40)
    thickness = random.randint(6, 10)
    x, y = random_position(length, length)
    draw.rectangle(
        [x, y + length // 2 - thickness // 2,
         x + length, y + length // 2 + thickness // 2],
        fill=255,
    )
    draw.rectangle(
        [x + length // 2 - thickness // 2, y,
         x + length // 2 + thickness // 2, y + length],
        fill=255,
    )


def generate_single_image(category):
    img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    draw = ImageDraw.Draw(img)
    {
        "circle": draw_circle,
        "square": draw_square,
        "triangle": draw_triangle,
        "rectangle": draw_rectangle,
        "cross": draw_cross,
    }[category](draw)
    return img


def active_dataset_categories():
    """Return all current ImageFolder class directories that contain PNG files."""
    if not RAW_DATASET_DIR.exists():
        return []
    active = []
    for folder in sorted(RAW_DATASET_DIR.iterdir()):
        if folder.is_dir() and any(folder.glob("*.png")):
            active.append(folder.name)
    return active


def count_dataset_images():
    return sum(
        len(list((RAW_DATASET_DIR / c).glob("*.png")))
        for c in active_dataset_categories()
    )


def rebuild_raw_dataset_dir():
    if RAW_DATASET_DIR.exists():
        shutil.rmtree(RAW_DATASET_DIR)
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)


def create_dataset(images_per_category, selected_categories):
    """Route A: create a dataset from the built-in geometric categories."""
    ensure_dirs()
    if not selected_categories:
        return "Please select at least one built-in shape category.", None

    rebuild_raw_dataset_dir()
    set_seed(DEFAULT_SEED)
    images_per_category = int(images_per_category)
    selected_categories = list(selected_categories)

    total_created = 0
    for category in selected_categories:
        category_dir = RAW_DATASET_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_category):
            image = generate_single_image(category)
            image.save(category_dir / f"{category}_{i:04d}.png")
            total_created += 1

    dataset_id = timestamp()
    config = {
        "dataset_id": dataset_id,
        "dataset_mode": "built_in_geometry",
        "image_size": IMAGE_SIZE,
        "images_per_category": images_per_category,
        "selected_categories": selected_categories,
        "total_images": total_created,
        "dataset_seed": DEFAULT_SEED,
        "controlled_variations": {
            "position_margin_px": 4,
            "size_range_px": [24, 44],
            "rotation": False,
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(DATASET_CONFIG_PATH, config)
    save_json(DATASET_OUTPUT_DIR / f"dataset_config_{dataset_id}.json", config)

    preview, preview_message = preview_dataset()
    message = (
        f"Built-in dataset created: {total_created} images across "
        f"{len(selected_categories)} categories.\n"
        f"Dataset ID: {dataset_id}\n"
        f"Training images: {project_relative(RAW_DATASET_DIR)}\n"
        f"{preview_message}"
    )
    return message, preview


def normalize_source_geometry(uploaded_image):
    """
    Convert an uploaded source image into the app's canonical representation:
    one white foreground geometry on a black background, 64x64 grayscale.
    The foreground is cropped, resized without changing aspect ratio, and centered.
    """
    if uploaded_image is None:
        raise ValueError("Upload a source geometry first.")

    gray = uploaded_image.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)
    binary = np.where(arr >= 128, 255, 0).astype(np.uint8)

    # If most pixels are white, treat white as the background and invert.
    if float(binary.mean()) > 127.5:
        binary = 255 - binary

    binary_img = Image.fromarray(binary, mode="L")
    bbox = binary_img.getbbox()
    if bbox is None:
        raise ValueError("No foreground geometry was detected in the uploaded image.")

    crop = binary_img.crop(bbox)
    target_long_side = 40
    scale = min(target_long_side / crop.width, target_long_side / crop.height)
    new_w = max(1, int(round(crop.width * scale)))
    new_h = max(1, int(round(crop.height * scale)))
    resampling = getattr(Image, "Resampling", Image).NEAREST
    crop = crop.resize((new_w, new_h), resample=resampling)

    canonical = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    x = (IMAGE_SIZE - new_w) // 2
    y = (IMAGE_SIZE - new_h) // 2
    canonical.paste(crop, (x, y))
    return canonical, crop


def generate_source_variation(source_crop):
    """Create one variation using only scale and position; no rotation."""
    target_long_side = random.randint(24, 44)
    scale = min(target_long_side / source_crop.width, target_long_side / source_crop.height)
    new_w = max(1, int(round(source_crop.width * scale)))
    new_h = max(1, int(round(source_crop.height * scale)))
    resampling = getattr(Image, "Resampling", Image).NEAREST
    resized = source_crop.resize((new_w, new_h), resample=resampling)

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    x, y = random_position(new_w, new_h)
    canvas.paste(resized, (x, y))
    return canvas


def create_student_source_dataset(uploaded_image, variations_count):
    """Route B: create a new dataset from one student-authored source geometry."""
    ensure_dirs()
    if uploaded_image is None:
        return "Upload a source geometry first.", None, None

    try:
        canonical, source_crop = normalize_source_geometry(uploaded_image)
    except Exception as exc:
        return f"Could not prepare the source geometry: {exc}", None, None

    rebuild_raw_dataset_dir()
    set_seed(DEFAULT_SEED)
    variations_count = int(variations_count)

    SOURCE_GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    canonical.save(SOURCE_GEOMETRY_PATH)
    source_history = SOURCE_GEOMETRY_DIR / f"source_geometry_{timestamp()}.png"
    canonical.save(source_history)

    class_dir = RAW_DATASET_DIR / CUSTOM_CLASS_NAME
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(variations_count):
        variation = generate_source_variation(source_crop)
        variation.save(class_dir / f"source_variation_{i:04d}.png")

    dataset_id = timestamp()
    config = {
        "dataset_id": dataset_id,
        "dataset_mode": "student_source_geometry",
        "image_size": IMAGE_SIZE,
        "source_geometry": project_relative(SOURCE_GEOMETRY_PATH),
        "total_images": variations_count,
        "dataset_seed": DEFAULT_SEED,
        "controlled_variations": {
            "position_margin_px": 4,
            "size_range_px": [24, 44],
            "rotation": False,
            "preserve_aspect_ratio": True,
        },
        "normalization": "grayscale -> threshold -> white foreground on black background -> crop -> centered canonical 64x64",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(DATASET_CONFIG_PATH, config)
    save_json(DATASET_OUTPUT_DIR / f"dataset_config_{dataset_id}.json", config)

    preview, preview_message = preview_dataset()
    message = (
        f"Student-source dataset created: {variations_count} controlled variations.\n"
        f"Only position and scale vary; rotation is not used.\n"
        f"Dataset ID: {dataset_id}\n"
        f"Normalized source saved to: {project_relative(SOURCE_GEOMETRY_PATH)}\n"
        f"{preview_message}"
    )
    return message, preview, canonical


def sample_images_from_folder(folder, n=5):
    files = list(Path(folder).glob("*.png"))
    if not files:
        return []
    if len(files) <= n:
        return files
    return random.sample(files, n)


def preview_dataset():
    ensure_dirs()
    categories = active_dataset_categories()
    if not categories:
        return None, "The training dataset is empty. Create a dataset first."

    set_seed(DEFAULT_SEED)
    cols = 5
    rows = len(categories)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1), squeeze=False)

    for i, category in enumerate(categories):
        sampled = sample_images_from_folder(RAW_DATASET_DIR / category, cols)
        for j in range(cols):
            ax = axes[i, j]
            if j >= len(sampled):
                ax.axis("off")
                continue
            with Image.open(sampled[j]) as img:
                ax.imshow(img.convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if j == 0:
                label = "Student source" if category == CUSTOM_CLASS_NAME else category.capitalize()
                ax.set_title(label)

    fig.suptitle("Training Dataset Preview", fontsize=14)
    fig.tight_layout()
    unique_path, latest_path = save_figure_history(fig, DATASET_OUTPUT_DIR, "dataset_preview")
    return open_pil_copy(latest_path), f"Dataset preview updated. Saved to: {project_relative(unique_path)}"



def reset_dataset():
    if RAW_DATASET_DIR.exists():
        shutil.rmtree(RAW_DATASET_DIR)
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_CONFIG_PATH.exists():
        DATASET_CONFIG_PATH.unlink()
    return "Current training dataset cleared. Previous output figures and source-geometry history were preserved.", None


# ============================================================
# MODELS
# ============================================================

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_enc(x)

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(x.size(0), 64, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(x.size(0), 64, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ============================================================
# DATALOADER
# ============================================================

def make_dataloader(batch_size):
    categories = active_dataset_categories()
    if not categories:
        return None, None, "The training dataset is empty. Create a dataset first."

    transform = base_transform()
    try:
        dataset = datasets.ImageFolder(root=str(RAW_DATASET_DIR), transform=transform)
    except Exception as exc:
        return None, None, f"Could not load the dataset: {exc}"

    if len(dataset) == 0:
        return None, None, "The training dataset is empty. Create a dataset first."

    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    return dataset, dataloader, None


# ============================================================
# AUTOENCODER
# ============================================================

def reconstruction_mse_per_sample(x_rec, x):
    """MSE summed over pixels per sample, then averaged over the batch."""
    per_pixel = F.mse_loss(x_rec, x, reduction="none")
    per_sample = per_pixel.flatten(1).sum(dim=1)
    return per_sample.mean()


def train_autoencoder(epochs, batch_size, latent_dim, learning_rate):
    ensure_dirs()
    set_seed(DEFAULT_SEED)

    dataset, dataloader, error = make_dataloader(batch_size)
    if error:
        return error, None

    epochs = int(epochs)
    batch_size = int(batch_size)
    latent_dim = int(latent_dim)
    learning_rate = float(learning_rate)

    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset_config = current_dataset_config()

    log_lines = [
        "AUTOENCODER TRAINING",
        f"Dataset ID: {dataset_config.get('dataset_id', 'n/a')}",
        f"Dataset mode: {dataset_config.get('dataset_mode', 'n/a')}",
        f"Training images: {len(dataset)}",
        f"Classes: {', '.join(active_dataset_categories())}",
        f"Epochs: {epochs}",
        f"Batch size: {batch_size}",
        f"Latent dimension: {latent_dim}",
        f"Learning rate: {learning_rate}",
        "Reconstruction loss: MSE summed over pixels per sample, averaged over batch",
        f"Seed: {DEFAULT_SEED}",
        f"Device: {compute_device_label()}",
        "",
    ]

    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = reconstruction_mse_per_sample(outputs, imgs)
            loss.backward()
            optimizer.step()
            b = imgs.size(0)
            running_loss += loss.item() * b
            seen += b
        epoch_loss = running_loss / max(1, seen)
        log_lines.append(f"Epoch {epoch + 1}/{epochs} - Reconstruction MSE: {epoch_loss:.4f}")
    training_seconds = time.perf_counter() - start_time

    torch.save(model.state_dict(), AUTOENCODER_MODEL_PATH)

    config = {
        "model_type": "Autoencoder",
        "epochs": epochs,
        "batch_size": batch_size,
        "latent_dim": latent_dim,
        "learning_rate": learning_rate,
        "training_images": len(dataset),
        "classes": active_dataset_categories(),
        "dataset_id": dataset_config.get("dataset_id"),
        "dataset_config": dataset_config,
        "seed": DEFAULT_SEED,
        "reconstruction_loss": "MSE summed over pixels per sample, averaged over batch",
        "training_seconds": round(training_seconds, 3),
        "runtime": runtime_metadata(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(AUTOENCODER_CONFIG_PATH, config)
    save_json(AE_TRAINING_DIR / f"autoencoder_config_{timestamp()}.json", config)

    model.eval()
    imgs, _ = next(iter(dataloader))
    imgs = imgs.to(device)
    with torch.no_grad():
        rec = model(imgs)
    imgs = imgs.cpu()
    rec = rec.cpu()

    n = min(8, imgs.size(0))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.7, 3.8), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(imgs[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(rec[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=10)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=10)
    fig.suptitle("Autoencoder Training Preview", fontsize=14)
    fig.tight_layout()
    unique_path, latest_path = save_figure_history(fig, AE_TRAINING_DIR, "training_preview")

    log_lines.append("")
    log_lines.append(f"Training time: {training_seconds:.2f} s")
    log_text = "\n".join(log_lines)
    save_text_history(log_text, AE_TRAINING_DIR, "training_log")
    log_text += f"\nModel saved to: {project_relative(AUTOENCODER_MODEL_PATH)}\n"
    log_text += f"Preview saved to: {project_relative(unique_path)}"
    return log_text, open_pil_copy(latest_path)

def load_autoencoder_model():
    if not AUTOENCODER_MODEL_PATH.exists():
        return None, "No trained Autoencoder model was found. Train the model first."
    config = load_json(AUTOENCODER_CONFIG_PATH)
    if config is None:
        return None, "Autoencoder configuration file is missing."
    latent_dim = int(config.get("latent_dim", 16))
    try:
        model = ConvAutoencoder(latent_dim=latent_dim).to(device)
        state_dict = torch.load(AUTOENCODER_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model, f"Loaded Autoencoder (latent dimension: {latent_dim})."
    except Exception as exc:
        return None, f"Could not load the Autoencoder: {exc}"


def run_autoencoder_reconstruction(uploaded_image):
    if uploaded_image is None:
        return None, "Upload a test image first."

    model, message = load_autoencoder_model()
    if model is None:
        return None, message

    original, x = prepare_input_image(uploaded_image)
    with torch.no_grad():
        rec = model(x)
    reconstructed = tensor_to_pil(rec)

    run_id = timestamp()
    original.save(AE_RECON_DIR / f"user_input_{run_id}.png")
    reconstructed.save(AE_RECON_DIR / f"reconstruction_{run_id}.png")
    save_latest_image(original, AE_RECON_DIR, "user_input")
    save_latest_image(reconstructed, AE_RECON_DIR, "reconstruction")
    ae_config = load_json(AUTOENCODER_CONFIG_PATH) or {}
    save_json(
        AE_RECON_DIR / "latest_reconstruction_metadata.json",
        {
            "operation": "ae_reconstruction",
            "input_hash": image_signature(original),
            "dataset_id": ae_config.get("dataset_id"),
            "model_trained_at": ae_config.get("trained_at"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    fig = make_side_by_side(original, reconstructed, "Post-training Input", "AE Reconstruction")
    unique_path, latest_path = save_figure_history(fig, AE_RECON_DIR, "comparison")

    return (
        open_pil_copy(latest_path),
        f"{message}\nReconstruction completed. The uploaded image was used only as a post-training input; "
        f"it was not added to the training dataset and did not change model weights.\n"
        f"Saved to: {project_relative(unique_path)}",
    )


def reset_autoencoder():
    if AUTOENCODER_MODEL_PATH.exists():
        AUTOENCODER_MODEL_PATH.unlink()
    if AUTOENCODER_CONFIG_PATH.exists():
        AUTOENCODER_CONFIG_PATH.unlink()
    return "Current Autoencoder model cleared. Previous output figures were preserved.", None, None


# ============================================================
# VAE
# ============================================================

def vae_loss(x_rec, x, mu, logvar):
    """
    Reconstruction: same per-sample summed MSE convention used by the AE.
    KL: summed across latent dimensions per sample, then averaged across batch.
    """
    recon_loss = reconstruction_mse_per_sample(x_rec, x)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_per_sample.mean()
    total = recon_loss + KL_WEIGHT * kl_loss
    return total, recon_loss, kl_loss


def train_vae(epochs, batch_size, latent_dim, learning_rate):
    ensure_dirs()
    set_seed(DEFAULT_SEED)

    dataset, dataloader, error = make_dataloader(batch_size)
    if error:
        return error, None

    epochs = int(epochs)
    batch_size = int(batch_size)
    latent_dim = int(latent_dim)
    learning_rate = float(learning_rate)

    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset_config = current_dataset_config()

    log_lines = [
        "VAE TRAINING",
        f"Dataset ID: {dataset_config.get('dataset_id', 'n/a')}",
        f"Dataset mode: {dataset_config.get('dataset_mode', 'n/a')}",
        f"Training images: {len(dataset)}",
        f"Classes: {', '.join(active_dataset_categories())}",
        f"Epochs: {epochs}",
        f"Batch size: {batch_size}",
        f"Latent dimension: {latent_dim}",
        f"Learning rate: {learning_rate}",
        "Reconstruction loss: MSE summed over pixels per sample, averaged over batch",
        f"KL weight: {KL_WEIGHT}",
        "KL scaling: summed over latent dimensions per sample, averaged over batch",
        f"Seed: {DEFAULT_SEED}",
        f"Device: {compute_device_label()}",
        "",
    ]

    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = total_recon = total_kl = 0.0
        seen = 0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            outputs, mu, logvar = model(imgs)
            loss, recon_loss, kl_loss = vae_loss(outputs, imgs, mu, logvar)
            loss.backward()
            optimizer.step()
            b = imgs.size(0)
            total_loss += loss.item() * b
            total_recon += recon_loss.item() * b
            total_kl += kl_loss.item() * b
            seen += b

        avg_loss = total_loss / max(1, seen)
        avg_recon = total_recon / max(1, seen)
        avg_kl = total_kl / max(1, seen)
        log_lines.append(
            f"Epoch {epoch + 1}/{epochs} - Total: {avg_loss:.4f} | "
            f"Reconstruction: {avg_recon:.4f} | KL: {avg_kl:.4f}"
        )
    training_seconds = time.perf_counter() - start_time

    torch.save(model.state_dict(), VAE_MODEL_PATH)

    config = {
        "model_type": "VAE",
        "epochs": epochs,
        "batch_size": batch_size,
        "latent_dim": latent_dim,
        "learning_rate": learning_rate,
        "training_images": len(dataset),
        "classes": active_dataset_categories(),
        "dataset_id": dataset_config.get("dataset_id"),
        "dataset_config": dataset_config,
        "seed": DEFAULT_SEED,
        "reconstruction_loss": "MSE summed over pixels per sample, averaged over batch",
        "kl_weight": KL_WEIGHT,
        "kl_scaling": "KL summed over latent dimensions per sample, averaged over batch",
        "training_seconds": round(training_seconds, 3),
        "runtime": runtime_metadata(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(VAE_CONFIG_PATH, config)
    save_json(VAE_TRAINING_DIR / f"vae_config_{timestamp()}.json", config)

    # Deterministic preview using the latent mean (mu).
    model.eval()
    imgs, _ = next(iter(dataloader))
    imgs = imgs.to(device)
    with torch.no_grad():
        mu, _ = model.encode(imgs)
        rec = model.decode(mu)
    imgs = imgs.cpu()
    rec = rec.cpu()

    n = min(8, imgs.size(0))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.7, 3.8), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(imgs[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(rec[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=10)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=10)
    fig.suptitle("VAE Training Preview", fontsize=14)
    fig.tight_layout()
    unique_path, latest_path = save_figure_history(fig, VAE_TRAINING_DIR, "training_preview")

    log_lines.append("")
    log_lines.append(f"Training time: {training_seconds:.2f} s")
    log_text = "\n".join(log_lines)
    save_text_history(log_text, VAE_TRAINING_DIR, "training_log")
    log_text += f"\nModel saved to: {project_relative(VAE_MODEL_PATH)}\n"
    log_text += f"Preview saved to: {project_relative(unique_path)}"
    return log_text, open_pil_copy(latest_path)

def load_vae_model():
    if not VAE_MODEL_PATH.exists():
        return None, "No trained VAE model was found. Train the model first."
    config = load_json(VAE_CONFIG_PATH)
    if config is None:
        return None, "VAE configuration file is missing."
    latent_dim = int(config.get("latent_dim", 16))
    try:
        model = ConvVAE(latent_dim=latent_dim).to(device)
        state_dict = torch.load(VAE_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model, f"Loaded VAE (latent dimension: {latent_dim})."
    except Exception as exc:
        return None, f"Could not load the VAE: {exc}"


def run_vae_reconstruction(uploaded_image):
    if uploaded_image is None:
        return None, "Upload a test image first."

    model, message = load_vae_model()
    if model is None:
        return None, message

    original, x = prepare_input_image(uploaded_image)

    # Use mu directly for a stable, repeatable reconstruction exercise.
    with torch.no_grad():
        mu, _ = model.encode(x)
        rec = model.decode(mu)
    reconstructed = tensor_to_pil(rec)

    run_id = timestamp()
    original.save(VAE_RECON_DIR / f"user_input_{run_id}.png")
    reconstructed.save(VAE_RECON_DIR / f"reconstruction_{run_id}.png")
    save_latest_image(original, VAE_RECON_DIR, "user_input")
    save_latest_image(reconstructed, VAE_RECON_DIR, "reconstruction")
    vae_config = load_json(VAE_CONFIG_PATH) or {}
    save_json(
        VAE_RECON_DIR / "latest_reconstruction_metadata.json",
        {
            "operation": "vae_reconstruction",
            "input_hash": image_signature(original),
            "dataset_id": vae_config.get("dataset_id"),
            "model_trained_at": vae_config.get("trained_at"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    fig = make_side_by_side(original, reconstructed, "Post-training Input", "VAE Reconstruction")
    unique_path, latest_path = save_figure_history(fig, VAE_RECON_DIR, "comparison")

    return (
        open_pil_copy(latest_path),
        f"{message}\nReconstruction completed using the VAE latent mean (mu) for a stable comparison. "
        f"The uploaded image was not added to the training dataset and did not change model weights.\n"
        f"Saved to: {project_relative(unique_path)}",
    )


def generate_vae_samples(samples_count):
    model, message = load_vae_model()
    if model is None:
        return None, message

    config = load_json(VAE_CONFIG_PATH) or {}
    latent_dim = int(config.get("latent_dim", 16))
    samples_count = int(samples_count)

    with torch.no_grad():
        z = torch.randn(samples_count, latent_dim, device=device)
        samples = model.decode(z).cpu()

    # Save individual generated samples for later review or export.
    for old in VAE_GENERATION_DIR.glob("latest_sample_*.png"):
        old.unlink()
    run_id = timestamp()
    for i in range(samples_count):
        pil = tensor_to_pil(samples[i])
        pil.save(VAE_GENERATION_DIR / f"generated_{run_id}_{i+1:02d}.png")
        pil.save(VAE_GENERATION_DIR / f"latest_sample_{i+1:02d}.png")

    cols = int(math.ceil(math.sqrt(samples_count)))
    rows = int(math.ceil(samples_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            if idx < samples_count:
                ax.imshow(samples[idx].squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            idx += 1

    fig.suptitle("VAE Generated Samples", fontsize=14)
    fig.tight_layout()
    unique_path, latest_path = save_figure_history(fig, VAE_GENERATION_DIR, "generated_samples")

    metadata = {
        "operation": "vae_generation",
        "samples_count": samples_count,
        "latent_sampling": "z ~ N(0,I)",
        "image_input": False,
        "vae_trained_at": config.get("trained_at"),
        "dataset_id": config.get("dataset_id"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(VAE_GENERATION_DIR / "latest_generation_metadata.json", metadata)

    return (
        open_pil_copy(latest_path),
        f"{message}\nGenerated {samples_count} samples from z ~ N(0,I) and the trained VAE decoder. "
        f"No image input was used.\nSaved to: {project_relative(unique_path)}",
    )


def interpolate_vae(image_a, image_b, steps):
    if image_a is None or image_b is None:
        return None, "Upload both Image A and Image B for interpolation."

    model, message = load_vae_model()
    if model is None:
        return None, message

    config = load_json(VAE_CONFIG_PATH) or {}
    original_a, x1 = prepare_input_image(image_a)
    original_b, x2 = prepare_input_image(image_b)
    steps = int(steps)

    with torch.no_grad():
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)
        decoded_tensors = []
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            decoded_tensors.append(model.decode(z).squeeze().cpu())

    # Save raw interpolation assets. The original endpoints remain separate
    # from the decoded interpolation endpoints.
    for old in VAE_INTERPOLATION_DIR.glob("latest_decoded_*.png"):
        old.unlink()
    save_latest_image(original_a, VAE_INTERPOLATION_DIR, "input_a")
    save_latest_image(original_b, VAE_INTERPOLATION_DIR, "input_b")
    run_id = timestamp()
    for i, tensor in enumerate(decoded_tensors):
        pil = tensor_to_pil(tensor)
        pil.save(VAE_INTERPOLATION_DIR / f"decoded_{run_id}_{i+1:02d}.png")
        pil.save(VAE_INTERPOLATION_DIR / f"latest_decoded_{i+1:02d}.png")

    fig, axes = plt.subplots(1, steps + 2, figsize=((steps + 2) * 1.6, 2.6), squeeze=False)
    axes[0, 0].imshow(original_a, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Image A")
    axes[0, 0].axis("off")

    for i, img in enumerate(decoded_tensors):
        axes[0, i + 1].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, i + 1].set_title(f"{i + 1}")
        axes[0, i + 1].axis("off")

    axes[0, -1].imshow(original_b, cmap="gray", vmin=0, vmax=255)
    axes[0, -1].set_title("Image B")
    axes[0, -1].axis("off")
    fig.suptitle("VAE Latent Interpolation", fontsize=14)
    fig.tight_layout()

    unique_path, latest_path = save_figure_history(fig, VAE_INTERPOLATION_DIR, "interpolation")
    metadata = {
        "operation": "vae_interpolation",
        "steps": steps,
        "path": "linear interpolation between mu_A and mu_B",
        "vae_trained_at": config.get("trained_at"),
        "dataset_id": config.get("dataset_id"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(VAE_INTERPOLATION_DIR / "latest_interpolation_metadata.json", metadata)

    return (
        open_pil_copy(latest_path),
        f"{message}\nInterpolated linearly between the latent means of Image A and Image B in {steps} decoded steps. "
        f"The displayed original A/B images are separate from the decoded endpoints.\n"
        f"Saved to: {project_relative(unique_path)}",
    )


def reset_vae():
    if VAE_MODEL_PATH.exists():
        VAE_MODEL_PATH.unlink()
    if VAE_CONFIG_PATH.exists():
        VAE_CONFIG_PATH.unlink()
    return "Current VAE model cleared. Previous output figures were preserved.", None, None, None, None


# ============================================================
# COMPARE
# ============================================================

def run_comparison_from_saved_reconstructions():
    """Compare the latest AE and VAE reconstructions without asking for another upload."""
    required = {
        "ae_input": AE_RECON_DIR / "latest_user_input.png",
        "ae_reconstruction": AE_RECON_DIR / "latest_reconstruction.png",
        "vae_input": VAE_RECON_DIR / "latest_user_input.png",
        "vae_reconstruction": VAE_RECON_DIR / "latest_reconstruction.png",
        "ae_metadata": AE_RECON_DIR / "latest_reconstruction_metadata.json",
        "vae_metadata": VAE_RECON_DIR / "latest_reconstruction_metadata.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        return None, (
            "Compare needs one completed reconstruction in each model tab. "
            "Run an Autoencoder reconstruction and a VAE reconstruction first, using the same post-training image."
        )

    ae_meta = load_json(required["ae_metadata"]) or {}
    vae_meta = load_json(required["vae_metadata"]) or {}
    ae_config = load_json(AUTOENCODER_CONFIG_PATH) or {}
    vae_config = load_json(VAE_CONFIG_PATH) or {}

    if ae_meta.get("input_hash") != vae_meta.get("input_hash"):
        return None, (
            "The latest AE and VAE reconstructions were created from different input images. "
            "Use the same post-training image in both reconstruction tabs and run both reconstructions again."
        )

    ae_dataset_id = ae_config.get("dataset_id")
    vae_dataset_id = vae_config.get("dataset_id")
    if ae_dataset_id and vae_dataset_id and ae_dataset_id != vae_dataset_id:
        return None, (
            "The current AE and VAE were trained on different datasets. "
            "Train both models on the same current training dataset before comparing them."
        )

    if ae_meta.get("model_trained_at") != ae_config.get("trained_at"):
        return None, (
            "The saved AE reconstruction belongs to an older Autoencoder training run. "
            "Run the AE reconstruction again with the current model."
        )
    if vae_meta.get("model_trained_at") != vae_config.get("trained_at"):
        return None, (
            "The saved VAE reconstruction belongs to an older VAE training run. "
            "Run the VAE reconstruction again with the current model."
        )

    original = open_pil_copy(required["ae_input"])
    ae_pil = open_pil_copy(required["ae_reconstruction"])
    vae_pil = open_pil_copy(required["vae_reconstruction"])

    run_id = timestamp()
    original.save(COMPARE_DIR / f"user_input_{run_id}.png")
    ae_pil.save(COMPARE_DIR / f"ae_reconstruction_{run_id}.png")
    vae_pil.save(COMPARE_DIR / f"vae_reconstruction_{run_id}.png")
    save_latest_image(original, COMPARE_DIR, "user_input")
    save_latest_image(ae_pil, COMPARE_DIR, "ae_reconstruction")
    save_latest_image(vae_pil, COMPARE_DIR, "vae_reconstruction")

    fig = make_three_panel(original, ae_pil, vae_pil)
    unique_path, latest_path = save_figure_history(fig, COMPARE_DIR, "ae_vae_comparison")

    summary = (
        "SAME POST-TRAINING INPUT, TWO SAVED RECONSTRUCTIONS\n\n"
        f"Dataset ID: {ae_dataset_id or vae_dataset_id or 'n/a'}\n\n"
        "Autoencoder\n"
        f"- Epochs: {ae_config.get('epochs', 'n/a')}\n"
        f"- Batch size: {ae_config.get('batch_size', 'n/a')}\n"
        f"- Latent dimension: {ae_config.get('latent_dim', 'n/a')}\n"
        f"- Learning rate: {ae_config.get('learning_rate', 'n/a')}\n\n"
        "VAE\n"
        f"- Epochs: {vae_config.get('epochs', 'n/a')}\n"
        f"- Batch size: {vae_config.get('batch_size', 'n/a')}\n"
        f"- Latent dimension: {vae_config.get('latent_dim', 'n/a')}\n"
        f"- Learning rate: {vae_config.get('learning_rate', 'n/a')}\n\n"
        "The comparison reuses the latest validated reconstruction outputs; no new image is uploaded here.\n"
        f"Saved to: {project_relative(unique_path)}"
    )
    return open_pil_copy(latest_path), summary


def reset_compare():
    return None, "Comparison view cleared. Saved comparison files were preserved."


# ============================================================
# USER INTERFACE
# ============================================================

ensure_dirs()

PARAMETER_HELP = """
### What do the training parameters mean?
- **Epochs** — the number of complete passes through the training dataset.
- **Batch Size** — the number of images processed before one model update.
- **Latent Dimension** — the size of the compressed internal representation learned by the model.
- **Learning Rate** — the size of each optimization step during training.

Changing these parameters can affect convergence, reconstruction quality, and the structure of the learned representation.
"""


CUSTOM_CSS = """
html, body, gradio-app, #root {
    background: #f5f3ee !important;
    margin: 0 !important;
    min-height: 100%;
}
body {
    min-height: 100vh;
}
.gradio-container {
    width: calc(100vw - 64px) !important;
    max-width: 1360px !important;
    min-height: 100vh;
    margin: 0 auto !important;
    padding: 24px 0 48px 0 !important;
    background: #f5f3ee !important;
}
@media (max-width: 900px) {
    .gradio-container {
        width: calc(100vw - 28px) !important;
        padding-top: 14px !important;
    }
}
#hero-wrap {
    background: linear-gradient(135deg, #f9f7f2 0%, #f0ece2 100%);
    border: 1px solid #d8d2c4;
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 12px 34px rgba(36, 42, 48, 0.13);
}
.hero-kicker {
    color: #3f4b5a;
    font-size: 12px;
    letter-spacing: 0.12em;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hero-title {
    color: #1f2428;
    font-size: 34px;
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 10px;
}
.hero-subtitle {
    color: #49535a;
    font-size: 16px;
    line-height: 1.6;
    max-width: 880px;
}
.hero-note {
    margin-top: 14px;
    background: #ffffff;
    border-left: 4px solid #7b8796;
    padding: 12px 14px;
    border-radius: 10px;
    color: #2a3238;
    line-height: 1.5;
}
.section-card {
    background: #ffffff;
    border: 1px solid #ddd7ca;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 9px 28px rgba(36, 42, 48, 0.11);
    margin-bottom: 16px;
}
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #1f2428;
    margin-bottom: 8px;
}
.section-text {
    color: #49535a;
    line-height: 1.65;
    margin-bottom: 14px;
}
.flow-wrap, .pedagogy-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: stretch;
}
.flow-step {
    flex: 1 1 160px;
    min-width: 160px;
    background: #fbfaf7;
    border: 1px solid #d7d1c3;
    border-radius: 14px;
    padding: 14px 14px 12px 14px;
    position: relative;
}
.flow-step .num {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #3f4b5a;
    margin-bottom: 8px;
}
.flow-step .head {
    font-size: 16px;
    font-weight: 700;
    color: #1f2428;
    margin-bottom: 6px;
}
.flow-step .desc {
    font-size: 13px;
    line-height: 1.5;
    color: #556068;
}
.flow-arrow {
    align-self: center;
    font-size: 24px;
    color: #7b877f;
    padding: 0 2px;
}
.mini-tag-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin-top: 8px;
}
.mini-tag {
    background: #eef1f5;
    color: #364555;
    border: 1px solid #d2d8e0;
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 13px;
    line-height: 1.45;
}
.ped-item {
    flex: 1 1 220px;
    min-width: 220px;
    background: #fbfaf7;
    border: 1px solid #d7d1c3;
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0 3px 10px rgba(36, 42, 48, 0.055);
}
.ped-item .head {
    font-size: 15px;
    font-weight: 700;
    color: #1f2428;
    margin-bottom: 6px;
}
.ped-item .desc {
    font-size: 13px;
    color: #58636a;
    line-height: 1.55;
}
.input-highlight {
    background: #eef1f5;
    border: 1px solid #d2d8e0;
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 10px;
    color: #364555;
    line-height: 1.55;
}
.gr-button-secondary {
    border-color: #bfc7c1 !important;
}
.gr-box, .gr-panel, .gr-form, .gr-accordion, .gr-image, .gr-textbox, .gr-checkboxgroup, .gr-slider, .gradio-tabs {
    border-radius: 14px !important;
}
.gr-tabitem {
    padding-top: 12px !important;
}
h1, h2, h3, h4 {
    color: #1f2428;
}
"""

HERO_HTML = """
<div id="hero-wrap">
  <div class="hero-kicker">LOCAL • HANDS-ON • EDUCATIONAL</div>
  <div class="hero-title">Mini GenAI Lab</div>
  <div class="hero-subtitle">
    A simplified local learning environment for architecture students to examine selected processes of image-based machine learning through dataset construction, training, reconstruction, generation, and interpolation.
  </div>
  <div class="hero-note">
    <strong>Core idea:</strong> the models are trained on a controlled dataset before any post-training operation is performed. <strong>Post-training images enter later</strong> for reconstruction, interpolation, or comparison; generation does not require an image input. These images are not automatically added to the training dataset.
  </div>
</div>
"""

WORKFLOW_HTML = """
<div class="section-card">
  <div class="section-title">How the application works</div>
  <div class="section-text">
    The application begins with a controlled training dataset. The Autoencoder and VAE are trained independently on that dataset. After training, reconstruction, generation, and interpolation are treated as distinct operations; Compare is an auxiliary view of AE and VAE reconstructions from the same input.
  </div>

  <div class="flow-wrap">
    <div class="flow-step">
      <div class="num">01</div>
      <div class="head">Build Dataset</div>
      <div class="desc">Create a controlled 64×64 grayscale training dataset either from built-in geometry categories or from one student source geometry with controlled variations.</div>
    </div>
    <div class="flow-arrow">→</div>
    <div class="flow-step">
      <div class="num">02</div>
      <div class="head">Train Models</div>
      <div class="desc">Train the Autoencoder and VAE. Students can adjust epochs, batch size, latent dimension, and learning rate.</div>
    </div>
  </div>

  <div class="input-highlight">
    <strong>After training, the workflow branches:</strong> reconstruction uses a post-training image; VAE generation samples from the prior and does not require an image input; interpolation uses two post-training images; Compare displays the latest AE and VAE reconstructions produced from the same input.
  </div>

  <div class="pedagogy-wrap" style="margin-top:12px;">
    <div class="ped-item">
      <div class="head">Reconstruction</div>
      <div class="desc"><strong>Post-training Image → Encoder → Latent Representation → Decoder → Reconstruction.</strong><br>The uploaded image is used only after training and is not added to the training dataset.</div>
    </div>
    <div class="ped-item">
      <div class="head">VAE Generation</div>
      <div class="desc"><strong>Random Latent Vector → VAE Decoder → Generated Sample.</strong><br>No new user image is required for this operation.</div>
    </div>
    <div class="ped-item">
      <div class="head">VAE Interpolation</div>
      <div class="desc"><strong>Image A + Image B → Encoded Latent Means → Intermediate Latent Vectors → Decoder → Decoded Sequence.</strong><br>A smooth visual sequence does not by itself imply semantic or architectural continuity.</div>
    </div>
    <div class="ped-item">
      <div class="head">AE / VAE Comparison</div>
      <div class="desc"><strong>Same Post-training Input → AE Reconstruction + VAE Reconstruction.</strong><br>Compare loads the latest saved reconstructions from the same input; it is not a ranking of model quality.</div>
    </div>
  </div>
</div>
"""

PEDAGOGY_HTML = """
<div class="section-card">
  <div class="section-title">What students can examine through the application</div>
  <div class="section-text">
    The platform is intentionally simplified so that students can connect model inputs, training choices, internal representations, and outputs instead of limiting the activity to the use of a pretrained image generator.
  </div>
  <div class="pedagogy-wrap">
    <div class="ped-item">
      <div class="head">1. Dataset Awareness</div>
      <div class="desc">Students can see that learning is conditioned by the selected training images and by the formal variation contained in the dataset.</div>
    </div>
    <div class="ped-item">
      <div class="head">2. Training Parameters</div>
      <div class="desc">Students adjust epochs, batch size, latent dimension, and learning rate while the interface explains what each parameter controls.</div>
    </div>
    <div class="ped-item">
      <div class="head">3. Reconstruction as a Learning Task</div>
      <div class="desc">The Autoencoder makes the sequence input → encoding → latent representation → decoding → reconstruction easier to follow.</div>
    </div>
    <div class="ped-item">
      <div class="head">4. Probabilistic Latent Representation</div>
      <div class="desc">The VAE introduces a probabilistic latent representation and shows how it supports reconstruction, random sampling, and interpolation.</div>
    </div>
    <div class="ped-item">
      <div class="head">5. Reconstruction, Generation, and Interpolation</div>
      <div class="desc">Students can distinguish between reconstructing a supplied input, generating samples from random latent vectors, and creating transitions between two encoded inputs.</div>
    </div>
    <div class="ped-item">
      <div class="head">6. Relating Outputs to Process</div>
      <div class="desc">The application prompts students to relate outputs to training data, model configuration, and the computational operation used, rather than treating generated images as autonomous results.</div>
    </div>
  </div>
  <div class="mini-tag-grid">
    <div class="mini-tag"><strong>For teaching:</strong> supports lectures, tutorials, and live demonstrations.</div>
    <div class="mini-tag"><strong>For students:</strong> provides a low-barrier introduction to image-based machine learning and generative models.</div>
    <div class="mini-tag"><strong>For experimentation:</strong> saves training configurations and output history so different runs can be reviewed and compared.</div>
  </div>
</div>
"""

GRADIO_MAJOR = int(str(gr.__version__).split(".")[0])
_blocks_kwargs = {"title": "Mini GenAI Lab"}
if GRADIO_MAJOR < 6:
    _blocks_kwargs["css"] = CUSTOM_CSS

with gr.Blocks(**_blocks_kwargs) as app:
    gr.HTML(HERO_HTML)

    with gr.Tab("0. Overview"):
        gr.HTML(WORKFLOW_HTML)
        gr.HTML(PEDAGOGY_HTML)
        gr.Markdown(
            """
### Suggested reading of the app
- Start with **Dataset** to define what the model will learn from.
- Continue to **Autoencoder** to understand reconstruction.
- Move to **Variational Autoencoder (VAE)** to explore generation and interpolation.
- Finish with **Compare** to inspect the latest AE and VAE reconstructions produced from the same post-training input.

All generated figures, logs, model configurations, and output images are saved locally in the project folders.
"""
        )

    with gr.Tab("1. Dataset"):
        gr.Markdown("## Stage 1 — Dataset Construction")
        gr.Markdown(
            "Choose **one** dataset route. Creating a dataset rebuilds the current training dataset from scratch, "
            "so the route used most recently is the material on which the next AE/VAE training run will learn."
        )

        gr.Markdown("### Route A — Built-in Geometry")
        gr.Markdown(
            "Generate a controlled 64×64 grayscale dataset from selected built-in categories. "
            "The controlled variation is position and size."
        )
        images_per_category = gr.Slider(
            minimum=10, maximum=500, value=200, step=10,
            label="Images per Category",
        )
        category_choices = gr.CheckboxGroup(
            choices=CATEGORIES,
            value=CATEGORIES,
            label="Built-in Shape Categories",
        )
        btn_create_dataset = gr.Button("Create Built-in Dataset", variant="primary")

        gr.Markdown("### Route B — Student Source Geometry")
        gr.Markdown(
            "Upload one source geometry. The application converts it to the common representation "
            "(white foreground on black background, 64×64) and builds a new dataset using **position and scale only**. "
            "Rotation is not used in this version."
        )
        with gr.Row():
            student_source_image = gr.Image(label="Student Source Geometry", type="pil")
            student_source_normalized = gr.Image(label="Normalized Source Geometry", type="pil")
        student_variations_count = gr.Slider(
            minimum=20, maximum=1000, value=200, step=10,
            label="Number of Controlled Variations",
        )
        btn_create_student_dataset = gr.Button("Create Dataset from Student Source", variant="primary")

        gr.Markdown("### Current Training Dataset")
        with gr.Row():
            btn_preview_dataset = gr.Button("Refresh Preview")
            btn_reset_dataset = gr.Button("Clear Current Dataset")
        dataset_message = gr.Textbox(label="Dataset Status", lines=6)
        dataset_preview = gr.Image(label="Training Dataset Preview", type="pil")

        gr.Markdown(
            "**Important distinction:** the dataset is training material. A post-training image used later for "
            "reconstruction, comparison, or interpolation is not automatically added to this dataset."
        )

        btn_create_dataset.click(
            fn=create_dataset,
            inputs=[images_per_category, category_choices],
            outputs=[dataset_message, dataset_preview],
        )
        btn_create_student_dataset.click(
            fn=create_student_source_dataset,
            inputs=[student_source_image, student_variations_count],
            outputs=[dataset_message, dataset_preview, student_source_normalized],
        )
        btn_preview_dataset.click(
            fn=preview_dataset,
            inputs=None,
            outputs=[dataset_preview, dataset_message],
        )
        btn_reset_dataset.click(
            fn=reset_dataset,
            inputs=None,
            outputs=[dataset_message, dataset_preview],
        )

    with gr.Tab("2. Autoencoder"):
        gr.Markdown("## Stage 2 — Training: Autoencoder")
        gr.Markdown(
            "The Autoencoder learns to compress an input image into a latent representation and then reconstruct it. "
            "It is used here as the introductory model for understanding **input → latent representation → reconstruction**."
        )

        with gr.Accordion("Training parameter guide", open=False):
            gr.Markdown(PARAMETER_HELP)

        ae_epochs = gr.Slider(1, 100, value=20, step=1, label="Epochs")
        ae_batch = gr.Slider(4, 128, value=32, step=4, label="Batch Size")
        ae_latent = gr.Slider(2, 64, value=16, step=2, label="Latent Dimension")
        ae_lr = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Learning Rate")

        with gr.Row():
            btn_train_ae = gr.Button("Train Autoencoder", variant="primary")
            btn_reset_ae = gr.Button("Clear Current Autoencoder")

        ae_log = gr.Textbox(label="Training Log", lines=12)
        ae_train_preview = gr.Image(label="Training Preview: Input vs Reconstruction", type="pil")

        gr.Markdown("### Stage 3 — Reconstruction with the Autoencoder")
        gr.Markdown(
            "**This is where a post-training image enters the reconstruction workflow.** The uploaded image is used only after training. "
            "It is **not** added to the dataset and does **not** retrain the model."
        )
        ae_test_image = gr.Image(label="Post-training Image", type="pil")
        btn_run_ae = gr.Button("Reconstruct with Autoencoder", variant="primary")
        ae_result = gr.Image(label="Autoencoder Reconstruction", type="pil")
        ae_result_msg = gr.Textbox(label="Reconstruction Status", lines=5)

        btn_train_ae.click(
            fn=train_autoencoder,
            inputs=[ae_epochs, ae_batch, ae_latent, ae_lr],
            outputs=[ae_log, ae_train_preview],
        )
        btn_run_ae.click(
            fn=run_autoencoder_reconstruction,
            inputs=ae_test_image,
            outputs=[ae_result, ae_result_msg],
        )
        btn_reset_ae.click(
            fn=reset_autoencoder,
            inputs=None,
            outputs=[ae_log, ae_train_preview, ae_result],
        )

    with gr.Tab("3. Variational Autoencoder (VAE)"):
        gr.Markdown("## Stage 2 — Training: Variational Autoencoder")
        gr.Markdown(
            "The VAE learns a probabilistic latent representation. In addition to reconstruction, "
            "it can decode randomly sampled latent vectors and interpolate between two encoded inputs."
        )

        with gr.Accordion("Training parameter guide", open=False):
            gr.Markdown(PARAMETER_HELP)
            gr.Markdown(
                "**VAE-specific idea:** training combines the same MSE-based reconstruction convention used by the AE with a KL-divergence term. "
                f"The KL weight is fixed at {KL_WEIGHT} and is not exposed as a learner control."
            )

        vae_epochs = gr.Slider(1, 100, value=30, step=1, label="Epochs")
        vae_batch = gr.Slider(4, 128, value=32, step=4, label="Batch Size")
        vae_latent = gr.Slider(2, 64, value=16, step=2, label="Latent Dimension")
        vae_lr = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Learning Rate")

        with gr.Row():
            btn_train_vae = gr.Button("Train VAE", variant="primary")
            btn_reset_vae = gr.Button("Clear Current VAE")

        vae_log = gr.Textbox(label="Training Log", lines=12)
        vae_train_preview = gr.Image(label="Training Preview: Input vs Reconstruction", type="pil")

        gr.Markdown("### Stage 3 — Reconstruction with the VAE")
        gr.Markdown(
            "The image is introduced **after training** and is used only for reconstruction. "
            "For a stable and repeatable result, reconstruction uses the VAE latent mean (mu), not a random sample."
        )
        vae_test_image = gr.Image(label="Post-training Image", type="pil")
        btn_run_vae_reconstruction = gr.Button("Reconstruct with VAE", variant="primary")
        vae_reconstruction_result = gr.Image(label="VAE Reconstruction", type="pil")
        vae_reconstruction_msg = gr.Textbox(label="Reconstruction Status", lines=5)

        gr.Markdown("### Stage 4 — Generation")
        gr.Markdown(
            "Random latent vectors are sampled and passed directly through the trained VAE decoder. "
            "This demonstrates generation without supplying a new input image."
        )
        vae_samples_count = gr.Slider(4, 25, value=16, step=1, label="Number of Generated Samples")
        btn_generate_vae = gr.Button("Generate Samples", variant="primary")
        vae_generation_result = gr.Image(label="Generated Samples", type="pil")
        vae_generation_msg = gr.Textbox(label="Generation Status", lines=4)

        gr.Markdown("### Stage 5 — Interpolation")
        gr.Markdown(
            "Upload two post-training images. The VAE encodes each image, linearly interpolates between their latent means, "
            "and decodes the intermediate latent positions into an image sequence. A smooth sequence does not by itself imply semantic or architectural continuity."
        )
        with gr.Row():
            interpolation_a = gr.Image(label="Image A", type="pil")
            interpolation_b = gr.Image(label="Image B", type="pil")
        interpolation_steps = gr.Slider(4, 16, value=10, step=1, label="Interpolation Steps")
        btn_interpolate_vae = gr.Button("Run Latent Interpolation", variant="primary")
        vae_interpolation_result = gr.Image(label="Latent Interpolation", type="pil")
        vae_interpolation_msg = gr.Textbox(label="Interpolation Status", lines=4)

        btn_train_vae.click(
            fn=train_vae,
            inputs=[vae_epochs, vae_batch, vae_latent, vae_lr],
            outputs=[vae_log, vae_train_preview],
        )
        btn_run_vae_reconstruction.click(
            fn=run_vae_reconstruction,
            inputs=vae_test_image,
            outputs=[vae_reconstruction_result, vae_reconstruction_msg],
        )
        btn_generate_vae.click(
            fn=generate_vae_samples,
            inputs=vae_samples_count,
            outputs=[vae_generation_result, vae_generation_msg],
        )
        btn_interpolate_vae.click(
            fn=interpolate_vae,
            inputs=[interpolation_a, interpolation_b, interpolation_steps],
            outputs=[vae_interpolation_result, vae_interpolation_msg],
        )
        btn_reset_vae.click(
            fn=reset_vae,
            inputs=None,
            outputs=[vae_log, vae_train_preview, vae_reconstruction_result, vae_generation_result, vae_interpolation_result],
        )

    with gr.Tab("4. Compare"):
        gr.Markdown("## Compare AE and VAE Reconstructions")
        gr.Markdown(
            "Run one reconstruction in the **Autoencoder** tab and one in the **VAE** tab using the same post-training image. "
            "Compare automatically loads those latest saved reconstructions, so you do not upload the image again here. "
            "If the two reconstructions came from different inputs, different datasets, or older model runs, the application will ask you to update them first."
        )
        with gr.Row():
            btn_run_compare = gr.Button("Compare Latest AE / VAE Reconstructions", variant="primary")
            btn_reset_compare = gr.Button("Clear Comparison View")
        compare_result = gr.Image(label="AE / VAE Reconstruction Comparison", type="pil", show_label=False)
        compare_text = gr.Textbox(label="Comparison Summary", lines=16)

        gr.Markdown(
            "**Important:** this comparison is pedagogical, not a ranking of which model is 'better'. "
            "It compares reconstruction behavior for the same input."
        )

        btn_run_compare.click(
            fn=run_comparison_from_saved_reconstructions,
            inputs=None,
            outputs=[compare_result, compare_text],
        )
        btn_reset_compare.click(
            fn=reset_compare,
            inputs=None,
            outputs=[compare_result, compare_text],
        )

    with gr.Tab("5. Local Files & Maintenance"):
        gr.Markdown("## Local Storage and Experiment Files")
        gr.Markdown(
            "This tab makes the local workflow explicit. The application stores datasets, trained models, "
            "and experiment outputs in folders next to `app.py`. Use **Open Outputs Folder** to inspect saved results. "
            "Use **Clear All Saved Outputs** only when you intentionally want to remove output history."
        )

        storage_info = gr.Textbox(
            label="Local Project Paths",
            value=storage_summary(),
            lines=6,
            interactive=False,
        )

        btn_refresh_storage = gr.Button("Refresh Local Paths")
        btn_open_outputs = gr.Button("Open Outputs Folder")
        storage_action_status = gr.Textbox(label="Storage Status", lines=5)

        gr.Markdown(
            "### Clear saved experiment outputs\n"
            "This deletes the contents of `outputs/`, including legacy files and experiment history. "
            "It does **not** delete the training dataset in `datasets/` or the trained models in `models/`. "
            "After clearing, the clean output folder structure is recreated automatically."
        )
        confirm_clear_outputs = gr.Checkbox(
            label="I understand that saved output images and logs will be deleted",
            value=False,
        )
        btn_clear_outputs = gr.Button("Clear All Saved Outputs")

        btn_refresh_storage.click(
            fn=storage_summary,
            inputs=None,
            outputs=storage_info,
        )
        btn_open_outputs.click(
            fn=open_outputs_folder,
            inputs=None,
            outputs=storage_action_status,
        )
        btn_clear_outputs.click(
            fn=clear_experiment_outputs,
            inputs=confirm_clear_outputs,
            outputs=storage_action_status,
        )


if __name__ == "__main__":
    print("Mini GenAI Lab v1")
    print(f"Project folder: {BASE_DIR}")
    print(f"Outputs folder: {OUTPUTS_DIR}")
    print("Local URL: http://127.0.0.1:7860")
    launch_kwargs = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "share": False,
        "inbrowser": True,
    }
    if GRADIO_MAJOR >= 6:
        launch_kwargs["css"] = CUSTOM_CSS
    app.launch(**launch_kwargs)
