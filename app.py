import streamlit as st
import torch
import segmentation_models_pytorch as smp
import numpy as np

from PIL import Image


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Waste Segmentation AI",
    page_icon="♻️",
    layout="wide"
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "best_waste_unet.pth"

IMAGE_SIZE = 256

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ============================================================
# Class Names
# ============================================================

CLASS_NAMES = [
    "background",
    "BIODEGRADABLE",
    "CARDBOARD",
    "GLASS",
    "METAL",
    "PAPER",
    "PLASTIC"
]


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    num_classes = checkpoint.get(
        "num_classes",
        len(CLASS_NAMES)
    )

    encoder_name = checkpoint.get(
        "encoder_name",
        "resnet34"
    )

    class_names = checkpoint.get(
        "class_names",
        CLASS_NAMES
    )

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
        activation=None
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, device, class_names


# ============================================================
# Image Preprocessing
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    original = image.copy()

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    image = (image - MEAN) / STD

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0)

    return original, tensor


# ============================================================
# Segmentation
# ============================================================

def predict(model, device, image):

    original, tensor = preprocess_image(image)

    tensor = tensor.to(device)

    with torch.no_grad():

        output = model(tensor)

        prediction = torch.argmax(
            output,
            dim=1
        )

    mask = prediction[0].cpu().numpy()

    return original, mask


# ============================================================
# Mask Visualization
# ============================================================

def create_color_mask(mask):

    h, w = mask.shape

    color_mask = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    # Background
    color_mask[mask == 0] = [0, 0, 0]

    # Biodegradable
    color_mask[mask == 1] = [46, 204, 113]

    # Cardboard
    color_mask[mask == 2] = [230, 126, 34]

    # Glass
    color_mask[mask == 3] = [52, 152, 219]

    # Metal
    color_mask[mask == 4] = [149, 165, 166]

    # Paper
    color_mask[mask == 5] = [241, 196, 15]

    # Plastic
    color_mask[mask == 6] = [231, 76, 60]

    return color_mask


# ============================================================
# Overlay
# ============================================================

def create_overlay(image, mask):

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_np = np.array(
        image.convert("RGB")
    )

    color_mask = create_color_mask(mask)

    overlay = image_np.copy()

    foreground = mask > 0

    overlay[foreground] = (
        0.6 * image_np[foreground]
        +
        0.4 * color_mask[foreground]
    ).astype(np.uint8)

    return overlay


# ============================================================
# Analyze Segmentation
# ============================================================

def analyze_mask(mask, class_names):

    total_pixels = mask.size

    waste_pixels = np.sum(mask > 0)

    waste_coverage = (
        waste_pixels / total_pixels
    ) * 100

    detected_classes = []

    class_percentages = {}

    for class_id in range(1, len(class_names)):

        pixels = np.sum(
            mask == class_id
        )

        percentage = (
            pixels / total_pixels
        ) * 100

        class_percentages[
            class_names[class_id]
        ] = percentage

        if pixels > 0:
            detected_classes.append(
                class_names[class_id]
            )

    if waste_pixels > 0:

        class_pixel_counts = {
            class_names[class_id]:
            np.sum(mask == class_id)
            for class_id in range(
                1,
                len(class_names)
            )
        }

        dominant_class = max(
            class_pixel_counts,
            key=class_pixel_counts.get
        )

    else:

        dominant_class = "None"

    return (
        waste_pixels > 0,
        waste_coverage,
        dominant_class,
        detected_classes,
        class_percentages
    )


# ============================================================
# Header
# ============================================================

st.title("♻️ Waste Segmentation AI")

st.markdown(
    """
    ### Multi-Class Waste Semantic Segmentation

    Upload an image and the U-Net model will identify and
    segment different types of waste.
    """
)

st.divider()


# ============================================================
# Load Model
# ============================================================

try:

    model, device, class_names = load_model()

except Exception as e:

    st.error(
        "Failed to load the model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.write(
        "**Architecture:** U-Net"
    )

    st.write(
        "**Encoder:** ResNet34"
    )

    st.write(
        "**Input Size:** 256 × 256"
    )

    st.write(
        f"**Classes:** {len(class_names) - 1}"
    )

    st.write(
        f"**Device:** {device}"
    )

    st.divider()

    st.subheader("Waste Classes")

    for class_name in class_names[1:]:

        st.write(
            f"• {class_name}"
        )


# ============================================================
# Upload Image
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# Main Application
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    with st.spinner(
        "Analyzing image..."
    ):

        original, mask = predict(
            model,
            device,
            image
        )

        overlay = create_overlay(
            original,
            mask
        )

        (
            waste_detected,
            waste_coverage,
            dominant_class,
            detected_classes,
            class_percentages
        ) = analyze_mask(
            mask,
            class_names
        )


    # ========================================================
    # Results
    # ========================================================

    st.subheader("Analysis Results")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        if waste_detected:

            st.metric(
                "Waste Detected",
                "YES"
            )

        else:

            st.metric(
                "Waste Detected",
                "NO"
            )


    with col2:

        st.metric(
            "Waste Coverage",
            f"{waste_coverage:.2f}%"
        )


    with col3:

        st.metric(
            "Dominant Waste",
            dominant_class
        )


    with col4:

        st.metric(
            "Detected Types",
            len(detected_classes)
        )


    st.divider()


    # ========================================================
    # Images
    # ========================================================

    st.subheader(
        "Segmentation Visualization"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.image(
            original,
            caption="Original Image",
            use_container_width=True
        )


    with col2:

        st.image(
            mask,
            caption="Segmentation Mask",
            use_container_width=True
        )


    with col3:

        st.image(
            overlay,
            caption="Segmentation Overlay",
            use_container_width=True
        )


    st.divider()


    # ========================================================
    # Detected Classes
    # ========================================================

    st.subheader(
        "Detected Waste Types"
    )


    if detected_classes:

        for class_name in detected_classes:

            percentage = class_percentages[
                class_name
            ]

            st.write(
                f"**{class_name}** — "
                f"{percentage:.2f}% of image"
            )

            st.progress(
                min(
                    percentage / 100,
                    1.0
                )
            )

    else:

        st.info(
            "No waste was detected in this image."
        )


    st.divider()


    # ========================================================
    # Summary
    # ========================================================

    st.subheader("Summary")

    if waste_detected:

        st.success(
            f"Waste detected. "
            f"The dominant type is "
            f"**{dominant_class}**, "
            f"covering approximately "
            f"**{waste_coverage:.2f}%** "
            f"of the image."
        )

    else:

        st.info(
            "The model did not detect any "
            "foreground waste in this image."
        )


else:

    st.info(
        "👆 Upload an image to start the analysis."
    )