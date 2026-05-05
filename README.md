# Construct a comprehensive text-to-image generating pipeline that includes GAN-based image generation, text preprocessing, and text embedding creation. This project simulates a real-world use case while integrating all the components.

# dataset for assignment 1:
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cifar, transform):
        self.data = cifar
        self.transform = transform

        self.labels_map = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        img = self.transform(img)
        text = f"a photo of a {self.labels_map[label]}"

        return img, text

# Methodology:
1. Text Processing:
Tokenization using BERT tokenizer
Embedding extraction (768-dim vector)
2. Generator:
Input: Noise + Text embedding
Fully connected layers
Output: 64×64 RGB image
3. Discriminator:
Input: Image + Text embedding
Output: Real / Fake classification
4. Training:
Loss: Binary Cross Entropy
Optimizer: Adam
Adversarial training (GAN)

# Result:
1. GAN Output:
Captures basic shapes and colors
Limited detail due to simple architecture
Diffusion Output

2. Using Stable Diffusion:
Highly realistic images
Better semantic alignment



###################################################################################################################################################################



# Use attention strategies like self-attention or cross-attention to improve a GAN. Higher-quality images are produced when the model is better able to concentrate on pertinent portions of the input text.

# Dataset for assignment 2:
dataset = load_dataset("cifar10")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def transform_fn(example):
    example["pixel_values"] = transform(example["img"])
    return example

dataset = dataset.with_transform(transform_fn)
train_dataset = dataset["train"]

from torch.utils.data import DataLoader
loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Methodology:
1. Text Encoding: Used CLIP to convert text into embeddings
2. Model: Conditional GAN with:
3. Generator: takes noise + text embedding
4. Discriminator: checks image-text consistency
5. Attention:
Cross-attention → aligns text with image features
Self-attention → focuses on important image regions
6. Training:
Loss: Binary Cross Entropy
Optimizer: Adam
Adversarial training (Generator vs Discriminator)


# Results:
GAN generated images captured basic shapes and object classes
1. Attention improved:
Better text-image alignment
More structured outputs
2.Still observed:
Blurry images
Limited detail (due to simple architecture + small dataset)
Compared with Stable Diffusion:
GAN → faster but lower quality
Diffusion → highly realistic images


###################################################################################################################################################################




# Use a custom dataset to refine a pre-trained text-to-image model (such as DALL-E or Stable Diffusion). This entails modifying the model to produce domain-specific visuals, such as artwork or medical imagery.

# Dataset for assignment 3:
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, prompt):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.prompt = prompt
        self.tokenizer = pipe.tokenizer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image.resize((512, 512))

        inputs = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        return {
            "pixel_values": torch.tensor(image).permute(2, 0, 1) / 255.0,
            "input_ids": inputs.input_ids[0]
        }

# Methodology:
1. Loaded pretrained Stable Diffusion model
2. Applied LoRA (Low-Rank Adaptation) using:
Rank = 8
Target layers: attention (to_q, to_k, to_v)
3. Used:
CLIP tokenizer for text encoding
UNet for image generation
4. Training Process:
Load custom images
Convert prompt → tokenized text
Pass image + text through diffusion model
Update only LoRA layers (not full model)
Save fine-tuned model
This enables efficient domain adaptation with low memory usage


#  Results:
Model successfully learns new concept (“sks_img2”)
1. Generated images reflect:
Custom style / object from dataset
Better alignment with prompt
2. Improvements:
More domain-specific outputs
Faster training (LoRA instead of full fine-tuning)
Lower GPU usage
3. Limitations:
Overfitting if dataset is very small
Quality depends on dataset diversity
Needs proper prompts for best results



######################################################################################################################################################################




# To comprehend the structure of a public dataset, load and examine it (e.g., COCO, Oxford-102 Flowers). Analyze dataset statistics such as the number of classes, description length, and image resolution, and explore and display text descriptions combined with photos.

# Dataset for assignment 4:
import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import Counter
from pycocotools.coco import COCO
if not os.path.exists("coco"):
    os.makedirs("coco/images", exist_ok=True)
    os.makedirs("coco/annotations", exist_ok=True)

    !wget -q http://images.cocodataset.org/zips/val2017.zip
    !wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip

    !unzip -q val2017.zip -d coco/images
    !unzip -q annotations_trainval2017.zip -d coco/
    ann_file = 'coco/annotations/captions_val2017.json'
coco = COCO(ann_file)

img_ids = coco.getImgIds()
all_ann_ids = coco.getAnnIds()

print("Total Images:", len(img_ids))
print("Total Captions:", len(all_ann_ids))
anns = coco.loadAnns(all_ann_ids)

caption_lengths = [len(a['caption'].split()) for a in anns]

print("\nCaption Statistics:")
print("Min:", np.min(caption_lengths))
print("Max:", np.max(caption_lengths))
print("Average:", np.mean(caption_lengths))

plt.hist(caption_lengths, bins=30)
plt.title("Caption Length Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()

    

# Methodology:
1. Data Loading
Used pycocotools to load annotations (captions_val2017.json)
Extracted image IDs and caption IDs
Caption Analysis
2. Calculated:
Minimum, maximum, and average caption length
Plotted histogram of caption lengths
Image Analysis
Extracted image width and height
Computed average resolution
Plotted scatter plot of width vs height
Text Analysis
3. Tokenized captions into words
Counted word frequency using Counter
Displayed top 20 most common words
4. Data Sampling & Visualization
Selected random images
Displayed images with corresponding captions
Saved sample dataset in JSON format



# Results:
1. Caption Statistics:
Shortest captions: ~1–2 words
Longest captions: ~15–20 words
Average length: ~10 words
2. Image Resolution:
Average width ≈ 480–640 px
Average height ≈ 480–640 px
Wide variation in resolution
3. Text Insights:
Most common words: “a”, “man”, “on”, “with”, “in”
Captions are simple, descriptive, and object-focused
4. Visualization Findings:
Each image has multiple captions → improves semantic richness
5. Captions describe:
Objects
Actions
Context (e.g., “man riding a bike”)



###################################################################################################################################################################




# Create a software that uses a library such as Hugging Face Transformers to preprocess text descriptions into tokenized and encoded representations. The text-to-image model will use these embeddings as inputs to make sure the text data is accurately represented


# Dataset for assignment 5:
prompts = [
    "A cat wearing sunglasses",
    "A robot playing guitar",
    "A fantasy castle in clouds"
]

for p in prompts:
    inputs = pipe.tokenizer(p, return_tensors="pt").to(device)
    emb = pipe.text_encoder(**inputs).last_hidden_state

    img = pipe(prompt_embeds=emb).images[0]

    plt.imshow(img)
    plt.title(p)
    plt.axis("off")
    plt.show()


# Methodology
1. Text Preprocessing
Used CLIP tokenizer
Converted text prompts into tokenized format
Text Encoding
Passed tokens into CLIP text encoder
2.Generated contextual embeddings (feature vectors)
Embedding Usage
Embeddings represent semantic meaning of text
These embeddings are fed into the diffusion model
Image Generation
3. Used Stable Diffusion pipeline
4. Generated images using:
prompt_embeds instead of raw text
This ensures precise control over input representation


# Results
1. The system successfully:
Converts text → embeddings
Generates images from embeddings
2. Generated images match prompt semantics:
“cat wearing sunglasses” → stylized cat image
“robot playing guitar” → mechanical figure with guitar
High-quality outputs due to pretrained diffusion model
3. Embedding-based input gives:
Better flexibility
More control than raw text prompts




###################################################################################################################################################################




# Create a CGAN that uses textual labels or categories to produce basic visuals. For instance, the model creates appropriate forms when labels like "square" or "circle" are provided. Here, conditional inputs in GANs are introduced

# Dataset for assignment 6:
class ShapeDataset(Dataset):
    def __init__(self, num_samples=2000, img_size=28):
        self.img_size = img_size
        self.data = []
        self.labels = []

        for _ in range(num_samples):
            label = np.random.randint(0, 2)
            img = np.zeros((img_size, img_size))

            if label == 0:
                cx, cy = img_size//2, img_size//2
                r = img_size//4
                for x in range(img_size):
                    for y in range(img_size):
                        if (x - cx)**2 + (y - cy)**2 < r**2:
                            img[x, y] = 1
            else:
                s = img_size//4
                img[img_size//2 - s:img_size//2 + s,
                    img_size//2 - s:img_size//2 + s] = 1

            self.data.append(img)
            self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = (img - 0.5) / 0.5
        return torch.tensor(img, dtype=torch.float32).view(-1), torch.tensor(self.labels[idx])



# Methodology
1. Conditional GAN (CGAN) usedGenerator: noise + label → image
2. Discriminator: image + label → real/fake
3. Label embeddings help control output
4  Trained using BCE loss and Adam optimizer

# Results
1. Model generates:
Label 0 → circle
Label 1 → square
Shapes become clearer after training
Shows successful label-controlled image generation


###################################################################################################################################################################
