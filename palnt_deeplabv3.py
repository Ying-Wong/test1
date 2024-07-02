import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained DeepLabV3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Define the image transformations
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
image_path = "01_colors_image_7.png"
input_image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB mode
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
input_batch = input_batch.to(device)

# Perform the forward pass to get the segmentation output
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# Define the colors for visualization
colors = np.random.randint(0, 255, size=(21, 3), dtype=np.uint8)

# Convert the predicted segmentation to an image
segmented_image = Image.fromarray(output_predictions.byte().cpu().numpy())
segmented_image.putpalette(colors)

# Plot the original and segmented images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(input_image)
ax1.set_title("Original Image")
ax1.axis('off')

ax2.imshow(segmented_image)
ax2.set_title("Segmented Image")
ax2.axis('off')

plt.show()
