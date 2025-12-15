import streamlit as st
import torch
import torch.nn as nn
from model_utils import RecipeTokenizer, RecipeTransformer, format_ingredients_for_input
import os
import re

# Set page configuration
st.set_page_config(
    page_title="NutriGen - AI Recipe Generator",
    page_icon="🍳",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #DFF2BF;
        border: 1px solid #4F8A10;
        color: #4F8A10;
        margin-top: 1rem;
    }
    .recipe-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🍳 NutriGen: AI-Powered Recipe Generator")
st.markdown("Generate personalized recipes based on your ingredients and dietary goals.")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Dietary Goals")
    
    target_calories = st.slider("Target Calories (kcal)", 200, 2000, 500, step=50)
    
    st.subheader("Macros Guide")
    target_protein = st.number_input("Protein (g)", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
    target_carbs = st.number_input("Carbs (g)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
    target_fat = st.number_input("Fat (g)", min_value=0.0, max_value=200.0, value=20.0, step=1.0)
    
    st.info("Adjust the sliders and inputs to match your nutritional requirements.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Ingredients")
    ingredients_input = st.text_area(
        "Enter available ingredients (separated by commas)",
        placeholder="e.g., chicken breast, broccoli, olive oil, garlic",
        height=200
    )
    
    generate_btn = st.button("✨ Generate Recipe")

# Model loading with caching
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Determine device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        st.write(f"Using device: {device}")
        
        # Load Tokenizer
        tokenizer_path = 'models/recipe_tokenizer.pkl'
        if not os.path.exists(tokenizer_path):
            st.error(f"Tokenizer not found at {tokenizer_path}")
            return None, None, None
            
        tokenizer = RecipeTokenizer.load(tokenizer_path)
        
        # Load Model
        model_path = 'models/nutrigen_transformer_final.pt'
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}")
            return None, None, None
            
        # Load checkpoint
        map_location = device
        checkpoint = torch.load(model_path, map_location=map_location)
        
        # Extract params from checkpoint if available, otherwise use defaults
        vocab_size = checkpoint.get('vocab_size', tokenizer.vocab_size)
        d_model = checkpoint.get('d_model', 256)
        # Note: Some checkpoints might not have all config params, so we keep defaults for safety
        # or inspection of 'config' key if it exists
        config = checkpoint.get('config', {})
        n_heads = config.get('n_heads', 8)
        n_layers = config.get('n_layers', 4)
        d_ff = config.get('d_ff', 1024)
        dropout = config.get('dropout', 0.1)
        
        st.write(f"Model config: d_model={d_model}, heads={n_heads}, layers={n_layers}")

        model = RecipeTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Handle DataParallel/Distributed wrapper if present
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Load resources
model, tokenizer, device = load_model_and_tokenizer()

# Inference logic
if generate_btn:
    if not ingredients_input:
        st.warning("Please enter some ingredients first!")
    elif model is None:
        st.error("Model failed to load. Please check the logs.")
    else:
        with col2:
            st.subheader("🍽️ Generated Recipe")
            with st.spinner("Chef AI is cooking..."):
                try:
                    # Parse ingredients
                    ingredients_list = [i.strip() for i in ingredients_input.split(',')]
                    
                    # Prepare nutrition info
                    nutrition_info = {
                        'calories': target_calories,
                        'protein': target_protein,
                        'carbs': target_carbs,
                        'fat': target_fat
                    }
                    
                    # Format input
                    input_text = format_ingredients_for_input(ingredients_list, nutrition_info)
                    
                    # Tokenize
                    input_ids = tokenizer.encode(input_text, max_length=200)
                    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
                    
                    # Generate
                    output_ids = model.generate(
                        input_tensor, 
                        tokenizer, 
                        max_length=300, 
                        temperature=0.8,
                        top_k=50
                    )
                    
                    # Decode
                    recipe_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                    
                    # Post-processing for display
                    # Simple formatting to capitalization
                    recipe_text = recipe_text.capitalize()
                    recipe_text = re.sub(r'([.!?])\s*([a-z])', lambda p: p.group(1) + " " + p.group(2).upper(), recipe_text)
                    
                    # Input Summary
                    st.info(f"**Based on:** {', '.join(ingredients_list)}\n\n**Goals:** {target_calories}kcal | P:{target_protein}g | C:{target_carbs}g | F:{target_fat}g")
                    
                    # Display Result
                    st.markdown(f"""
                    <div class="recipe-card">
                        <h3>Chef's Suggestion</h3>
                        <p style="font-size: 1.1em; line-height: 1.6;">{recipe_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred during generation: {str(e)}")
                    st.exception(e)

