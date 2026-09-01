# ♻️ Waste Semantic Segmentation Using U-Net

A multi-class semantic segmentation system for detecting and segmenting different types of waste using a U-Net architecture with a ResNet34 encoder.

The project covers the complete pipeline from dataset preprocessing and model training to evaluation and deployment using Streamlit.

## 🚀 Demo

The trained model is deployed as an interactive Streamlit application.

The application allows users to upload an image and automatically:

- Detect whether waste is present
- Segment waste regions
- Identify detected waste types
- Estimate waste coverage
- Determine the dominant waste type
- Visualize the segmentation mask
- Generate an overlay between the original image and segmentation result

## 🧠 Model

- Architecture: U-Net
- Encoder: ResNet34
- Encoder Weights: ImageNet pretrained
- Input Size: 256 × 256
- Task: Multi-Class Semantic Segmentation
- Framework: PyTorch
- Segmentation Library: Segmentation Models PyTorch

## 🗂️ Waste Classes

The model predicts 6 waste categories plus background:

- Biodegradable
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Background

## 📊 Evaluation

The model was evaluated using:

- Dice Score
- Mean Intersection over Union (mIoU)
- Per-class Dice and IoU

### Test Results

| Metric | Score |
|---|---:|
| Mean Dice | 0.469 |
| mIoU | 0.412 |

### Per-Class Performance

| Class | Dice | IoU |
|---|---:|---:|
| Biodegradable | 0.000 | 0.000 |
| Cardboard | 0.160 | 0.087 |
| Glass | 0.000 | 0.000 |
| Metal | 0.867 | 0.765 |
| Paper | 0.925 | 0.861 |
| Plastic | 0.865 | 0.761 |

The model performs strongly on Paper, Plastic, and Metal, while Glass, Biodegradable, and Cardboard remain more challenging classes.

## 🏗️ Pipeline

```text
Waste Dataset
      ↓
Image & Mask Preprocessing
      ↓
Image Resizing & Normalization
      ↓
Data Augmentation
      ↓
U-Net + ResNet34
      ↓
Training
      ↓
Validation
      ↓
Best Model Selection
      ↓
Test Evaluation
      ↓
Dice & mIoU
      ↓
Streamlit Deployment

💻 Project Structure

waste-semantic-segmentation-unet/
│
├── app.py
├── best_waste_unet.pth
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/waste-semantic-segmentation-unet.git

Navigate to the project directory:

cd waste-semantic-segmentation-unet

Install the required dependencies:

pip install -r requirements.txt

▶️ Run the Application

Launch the Streamlit application:

streamlit run app.py

The application will open in your browser.

📷 Application Features

The Streamlit application provides:

Waste Detection

Determines whether the uploaded image contains detectable waste.

Waste Coverage

Estimates the percentage of the image covered by detected waste.

Dominant Waste

Identifies the waste category occupying the largest segmented area.

Multi-Class Segmentation

Segments different waste categories independently.

Visualization

Displays:

Original Image

Segmentation Mask

Segmentation Overlay


📚 Dataset

The project uses a semantic segmentation waste dataset containing six waste categories:

Paper

Plastic

Glass

Metal

Cardboard

Biodegradable


The dataset was exported in semantic segmentation mask format.

🔬 Limitations

The model does not perform equally well across all classes.

Paper, Plastic, and Metal achieve strong segmentation performance, while Glass, Biodegradable, and Cardboard require further improvement.

Potential future improvements include:

Class balancing

Stronger data augmentation

Class-weighted or focal loss

Longer training

Alternative encoders

Higher input resolution

Additional training data


🔮 Future Work

Possible extensions include:

Improved class balancing

Real-time waste segmentation

Mobile deployment

Edge-device deployment

Improved segmentation architectures

Waste sorting assistance systems


🛠️ Technologies

Python

PyTorch

Segmentation Models PyTorch

NumPy

Pillow

Streamlit

Kaggle


👤 Author

Mohamed Adel Yousef Wetwet


---

⭐ If you find this project useful, feel free to star the repository.
