import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

# Load the EfficientNetB0 model pre-trained on ImageNet
model = EfficientNetB0(weights='imagenet')

# Plant library with categories
# Plant library with categories relevant to Morocco
plant_library = {
    "Spices": {
        "Ginger": {
            "scientific_name": "Zingiber officinale",
            "image": "img/ginger.jpeg",
            "description": "Medicinal properties for digestion and inflammation."
        },
        "Saffron": {
            "scientific_name": "Crocus sativus",
            "image": "img/saffron.jpeg",
            "description": "Known for its distinctive color and health benefits."
        },
        "Turmeric": {
            "scientific_name": "Curcuma longa",
            "image": "img/turmeric.jpeg",
            "description": "Anti-inflammatory properties."
        },
        "Cinnamon": {
            "scientific_name": "Cinnamomum",
            "image": "img/cinnamon.jpeg",
            "description": "Cinnamon is a spice known for its flavor and medicinal properties."
        },
        "Pepper": {
            "scientific_name": "Piper nigrum",
            "image": "img/pepper.jpeg",
            "description": "Black pepper is commonly used in Moroccan spices blends like Ras el Hanout."
        }
    },
    "Herbs": {
        "Mint": {
            "scientific_name": "Mentha",
            "image": "img/mint.jpeg",
            "description": "Mint is a fundamental herb in Moroccan tea and is known for its refreshing and digestive properties."
        },
        "Thyme": {
            "scientific_name": "Thymus",
            "image": "img/thyme.jpeg",
            "description": "Thyme is used in Moroccan cuisine and as a remedy for respiratory conditions."
        },
        "Basil": {
            "scientific_name": "Ocimum basilicum",
            "image": "img/basil.jpeg",
            "description": "Basil is used in Moroccan cooking and is known for its anti-inflammatory properties."
        },
        "Rosemary": {
            "scientific_name": "Salvia rosmarinus",
            "image": "img/rosemary.jpeg",
            "description": "Rosemary is a fragrant herb used in Moroccan cooking and traditional medicine."
        }
    },
    "Medicinal Plants": {
        "Aloe Vera": {
            "scientific_name": "Aloe barbadensis miller",
            "image": "img/aloe_vera.jpeg",
            "description": "Aloe Vera is known for its medicinal properties, particularly in treating skin conditions."
        },
        "Lavender": {
            "scientific_name": "Lavandula",
            "image": "img/lavender.jpeg",
            "description": "Lavender is used for anxiety, stress, and insomnia."
        },
        "Coriander": {
            "scientific_name": "Coriandrum sativum",
            "image": "img/coriander.jpeg",
            "description": "Coriander is commonly used in Moroccan cuisine and traditional medicine for digestive issues."
        }
    }
}

# Sidebar for Plant Library
st.sidebar.title("Plant Library")

# Category selection
categories = list(plant_library.keys())
selected_category = st.sidebar.selectbox("Choose a category", categories)

if selected_category:
    # Plant selection within the chosen category
    plants_in_category = plant_library[selected_category].keys()
    selected_plant = st.sidebar.selectbox(f"Choose a plant from {selected_category}", plants_in_category)
    
    if selected_plant:
        # Display plant information
        plant_info = plant_library[selected_category][selected_plant]
        st.sidebar.image(plant_info["image"], caption=selected_plant, use_column_width=True)
        st.sidebar.write(f"**Scientific Name:** {plant_info['scientific_name']}")
        st.sidebar.write(f"**Description:** {plant_info['description']}")

# Title of the Streamlit app
st.title("Plant Species Identification")

# File uploader for the image
uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image for EfficientNetB0
    img = img.resize((224, 224))  # Resize to the input size of the model
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Decode the results into a list of tuples (class, description, probability)
    results = decode_predictions(predictions, top=3)[0]

    # Display the top predictions
    st.write("Top Predictions:")
    identified_plant = None
    for i, (imagenet_id, label, score) in enumerate(results):
        st.write(f"{i + 1}. {label}: {score:.4f}")
        if i == 0:
            identified_plant = label  # Use the top prediction as the identified plant

    # Display information about the identified plant
    if identified_plant:
        st.write(f"## Identified Plant: {identified_plant}")
        # Check across all categories for the identified plant
        for category in plant_library.values():
            if identified_plant in category:
                plant_info = category[identified_plant]
                st.image(plant_info["image"], caption=identified_plant, use_column_width=True)
                st.write(f"**Scientific Name:** {plant_info['scientific_name']}")
                st.write(f"**Description:** {plant_info['description']}")
                break
        else:
            st.write("Information about this plant is not available in the library.")
