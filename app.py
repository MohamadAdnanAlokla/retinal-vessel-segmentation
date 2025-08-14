import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import gradio as gr
from PIL import Image

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# =========================
# 1. U-Net building blocks
# =========================
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs); x = self.bn1(x); x = self.relu(x)
        x = self.conv2(x); x = self.bn2(x); x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.b = conv_block(512, 1024)
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        self.outputs = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        outputs = self.outputs(d4)
        return outputs

# =========================
# 2. Load the trained model
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_unet().to(device)

# Model path relative to app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "unet_best.pth")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =========================
# 3. Prediction function (returns only mask)
# =========================
def predict_mask(input_img):
    # Convert image to NumPy array
    image = np.array(input_img)
    # Convert from RGB to BGR (as used during training)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = image_bgr.shape[:2]

    # Resize to 512x512
    image_resized = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    image_norm = image_resized / 255.0

    # Change channel order and add batch dimension
    tensor_img = torch.tensor(image_norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor_img))[0, 0].cpu().numpy()

    # Convert probabilities to binary mask
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return mask_resized  # Return only mask

# =========================
# 4. Gradio interface (mask only)
# =========================
interface = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="pil", label="Upload Retinal Image"),
    outputs=gr.Image(type="numpy", label="Predicted Mask"),
    title="Retinal Blood Vessel Segmentation",
    description="Upload a retinal image to extract blood vessels using the trained model."
)

if __name__ == "__main__":
    interface.launch(server_port=7860, inbrowser=True, share=False)
    input("Press Enter to exit...")
