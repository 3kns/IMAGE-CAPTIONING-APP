import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

#LOAD THE PRETRAINED PROCESSOR AND MODEL
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#LOADING MY IMAGE
img_path = r"C:\Users\ocran\Desktop\IMAGE_CAPTIONING\ojoo.jpeg"

#CONVERTING TO RGB FORMAT
image = Image.open(img_path).convert('RGB')

#YOU DO NOT NEED A QUESTION FOR IMAGE CAPTIONING
text = "the image of"
inputs = processor(images=image, text=text, return_tensors='pt')

#GENERATING CAPTIONS FOR THE IMAGE
outputs = model.generate(**inputs, max_length=50)

#DECODING GENERATED TOKENS TO TEXT
caption = processor.decode(outputs[0], skip_special_tokens=True)

#PRINTING THE CAPTION
print(caption)