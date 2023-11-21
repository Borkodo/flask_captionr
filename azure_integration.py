import requests
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from io import BytesIO

import os

region = "eastus"
key = "MY_KEY1234567"

credentials = CognitiveServicesCredentials(key)
client = ComputerVisionClient(
    endpoint="https://" + region + ".api.cognitive.microsoft.com/",
    credentials=credentials
)


def generate_caption(description):
    """
    Generates a caption based on a description.
    """
    # Initialize the GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(
        "captionr/fine_tuned_model")
    model = GPT2LMHeadModel.from_pretrained(
        "captionr/fine_tuned_model")

    input_text = f"{description}"

    # Encode the input string to tensor
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.8,
        no_repeat_ngram_size=1,
        do_sample=True
    )

    # Decode and return the generated text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text


def get_caption(image_path):
    """
    Retrieves caption for an image using Azure's Computer Vision API.
    """
    with open(image_path, 'rb') as img_f:
        img_data = img_f.read()

    img_stream = BytesIO(img_data)

    # Get description from Azure Computer Vision
    image_analysis = client.analyze_image_in_stream(img_stream, visual_features=[VisualFeatureTypes.description])
    description = image_analysis.description.captions[
        0].text if image_analysis.description.captions else "no description"

    # Generate caption using GPT-2
    caption = generate_caption(description)
    return caption
