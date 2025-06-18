from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import pickle
import json
from tensorflow.keras.applications.resnet import preprocess_input
import tempfile
from typing import Dict, List, Optional
from pydantic import BaseModel
import uvicorn
from pathlib import Path

app = FastAPI(title="Fashion Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
label_binarizers = None
num_classes = None
metadata_df = None

MODEL_PATH = "converted_model"
METADATA_CSV_PATH = "selected_metadata.csv"
IMAGES_FOLDER = "selected_images"  

app.mount("/images", StaticFiles(directory=IMAGES_FOLDER), name="images")

class PredictionResponse(BaseModel):
    input_metadata: Dict
    recommendations: Dict
    message: str

class HealthResponse(BaseModel):
    status: str
    message: str

def _load_model(model_path="converted_model"):
    """Load the fashion classification model and components."""
    model = tf.saved_model.load(model_path)        
    with open(os.path.join(model_path, 'label_binarizers.pkl'), 'rb') as f:
        label_binarizers = pickle.load(f)
        
    with open(os.path.join(model_path, 'num_classes.json'), 'r') as f:
        num_classes = json.load(f)
        
    return model, label_binarizers, num_classes

def load_fashion_metadata(csv_path):
    """Load fashion metadata from CSV file"""
    return pd.read_csv(csv_path)

def get_local_image_url(image_id, base_url="http://localhost:8000"):
    """Generate local image URL"""
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    for ext in extensions:
        image_filename = f"{image_id}{ext}"
        image_path = os.path.join(IMAGES_FOLDER, image_filename)
        
        if os.path.exists(image_path):
            return f"{base_url}/images/{image_filename}"
    
    return None

def enhance_with_image_urls(recommendations, base_url="http://localhost:8000"):
    """Add local image URLs to recommendations"""
    for category, items in recommendations.items():
        for item in items:
            image_id = str(item.get('id', ''))
            if image_id:
                item['image_url'] = get_local_image_url(image_id, base_url)
            else:
                item['image_url'] = None
    return recommendations

def get_fashion_recommendations(input_metadata, metadata_df, num_recommendations=3):
    """Get fashion recommendations based on input metadata"""
    input_gender = input_metadata.get('gender', '')
    input_category = input_metadata.get('subCategory', '')
    input_color = input_metadata.get('baseColour', '')
    input_season = input_metadata.get('season', '')
    input_usage = input_metadata.get('usage', '')
    
    complementary_mapping = {
        'Topwear': ['Bottomwear', 'Shoes', 'Watches', 'Accessories', 'Belts', 'Bags'],
        'Bottomwear': ['Topwear', 'Shoes', 'Watches', 'Belts', 'Accessories', 'Bags'],
        'Shoes': ['Topwear', 'Bottomwear', 'Watches', 'Accessories', 'Belts', 'Bags'],
        'Watches': ['Topwear', 'Bottomwear', 'Shoes', 'Accessories'],
        'Accessories': ['Topwear', 'Bottomwear', 'Shoes', 'Watches'],
        'Belts': ['Topwear', 'Bottomwear', 'Shoes'],
        'Bags': ['Topwear', 'Bottomwear', 'Shoes', 'Accessories'],
        'Sandal': ['Topwear', 'Bottomwear', 'Watches', 'Accessories', 'Bags'],
        'Flip Flops': ['Topwear', 'Bottomwear', 'Watches', 'Accessories', 'Bags'],
        'Casual Shoes': ['Topwear', 'Bottomwear', 'Watches', 'Accessories', 'Bags'],
        'Formal Shoes': ['Topwear', 'Bottomwear', 'Watches', 'Accessories'],
        'Sports Shoes': ['Topwear', 'Bottomwear', 'Accessories', 'Bags'],
    }
    
    complementary_categories = complementary_mapping.get(input_category, [])
    
    if not complementary_categories:
        all_categories = metadata_df['subCategory'].unique().tolist()
        complementary_categories = [cat for cat in all_categories if cat != input_category]
    
    if input_category in complementary_categories:
        complementary_categories.remove(input_category)
    
    color_compatibility = {
        'Red': ['Black', 'White', 'Grey', 'Blue', 'Navy Blue', 'Beige', 'Cream'],
        'Blue': ['White', 'Grey', 'Black', 'Beige', 'Navy Blue', 'Red', 'Cream'],
        'Black': ['White', 'Grey', 'Red', 'Blue', 'Green', 'Yellow', 'Pink', 'Purple', 'Silver'],
        'White': ['Black', 'Blue', 'Red', 'Grey', 'Navy Blue', 'Green', 'Purple', 'Pink'],
        'Grey': ['Black', 'White', 'Blue', 'Red', 'Navy Blue', 'Purple'],
        'Navy Blue': ['White', 'Grey', 'Red', 'Beige', 'Cream'],
        'Green': ['White', 'Black', 'Beige', 'Grey', 'Brown', 'Cream'],
        'Beige': ['Navy Blue', 'Blue', 'Brown', 'Black', 'White', 'Red', 'Green'],
        'Brown': ['Beige', 'White', 'Blue', 'Green', 'Cream', 'Tan'],
        'Yellow': ['Black', 'Blue', 'White', 'Navy Blue'],
        'Pink': ['Black', 'White', 'Grey', 'Navy Blue', 'Blue'],
        'Purple': ['White', 'Black', 'Grey', 'Silver'],
        'Orange': ['Black', 'White', 'Blue', 'Navy Blue'],
        'Silver': ['Black', 'White', 'Grey', 'Blue', 'Purple'],
        'Gold': ['Black', 'White', 'Brown', 'Beige'],
        'Cream': ['Black', 'Brown', 'Beige', 'Blue', 'Green'],
        'Tan': ['White', 'Black', 'Brown', 'Blue'],
        'Maroon': ['White', 'Black', 'Grey', 'Beige'],
        'Olive': ['White', 'Black', 'Beige', 'Brown'],
        'Turquoise Blue': ['White', 'Black', 'Grey', 'Beige'],
        'Multi': ['Black', 'White', 'Grey']  
    }
    
    compatible_colors = color_compatibility.get(input_color, [])
    
    if not compatible_colors:
        compatible_colors = ['Black', 'White', 'Grey', 'Navy Blue', 'Blue', 'Beige']
    
    if input_color not in compatible_colors:
        compatible_colors.append(input_color)
    
    filtered_df = metadata_df[metadata_df['gender'] == input_gender]
    
    recommendations = {}
    
    for category in complementary_categories:
        category_df = filtered_df[filtered_df['subCategory'] == category]
        
        if len(category_df) == 0:
            continue
        
        color_filter = category_df['baseColour'].isin(compatible_colors)
        season_filter = category_df['season'] == input_season
        usage_filter = category_df['usage'] == input_usage
        
        filter_combinations = [
            season_filter & usage_filter & color_filter,
            season_filter & color_filter,
            usage_filter & color_filter,
            color_filter,
            pd.Series([True] * len(category_df), index=category_df.index)
        ]
        
        matching_items = pd.DataFrame()
        
        for filter_combo in filter_combinations:
            matching_items = category_df[filter_combo]
            if len(matching_items) >= num_recommendations:
                break
        
        if len(matching_items) > 0:
            matching_items = matching_items.sample(n=min(len(matching_items), num_recommendations * 2))
            recommendations[category] = matching_items.head(num_recommendations).to_dict('records')
    
    return recommendations

def load_image(image_path):
    """Load and preprocess image"""
    IMAGE_DIMS = (180, 180, 3)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    return image

def predict_fashion_item(model, label_binarizers, image_array):
    """Make predictions using SavedModel format"""
    input_tensor = tf.convert_to_tensor(image_array[np.newaxis, ...], dtype=tf.float32)
    
    infer = model.signatures["serving_default"]
    predictions = infer(input_tensor)
    
    predictions_dict = {}
    for category, lb in label_binarizers.items():
        output_name = f"{category}_output"
        probs = predictions[output_name].numpy()[0]
        label_idx = np.argmax(probs)
        
        predictions_dict[category] = {
            'label': lb.classes_[label_idx],
            'confidence': float(probs[label_idx])
        }
    
    return predictions_dict

@app.on_event("startup")
async def startup_event():
    """Load model and metadata on startup"""
    global model, label_binarizers, num_classes, metadata_df
    
    try:
        print("Loading model...")
        model, label_binarizers, num_classes = _load_model(MODEL_PATH)
        print("Model loaded successfully")
        
        print("Loading metadata...")
        if not os.path.exists(METADATA_CSV_PATH):
            raise FileNotFoundError(f"Metadata file not found: {METADATA_CSV_PATH}")
        
        metadata_df = load_fashion_metadata(METADATA_CSV_PATH)
        print(f"Metadata loaded: {len(metadata_df)} items")
        
        if os.path.exists(IMAGES_FOLDER):
            image_count = len([f for f in os.listdir(IMAGES_FOLDER) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            print(f"Found {image_count} images in {IMAGES_FOLDER} folder")
        else:
            print(f"Warning: Images folder '{IMAGES_FOLDER}' not found")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise e

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(status="healthy", message="Fashion Recommendation API is running")


async def health_check():
    """Detailed health check"""
    global model, metadata_df
    
    if model is None or metadata_df is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    image_count @app.get("/health", response_model=HealthResponse)==0
    if os.path.exists(IMAGES_FOLDER):
        image_count = len([f for f in os.listdir(IMAGES_FOLDER) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    
    return HealthResponse(
        status="healthy", 
        message=f"API is ready. Dataset: {len(metadata_df)} items, Images: {image_count}"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_outfit(
    file: UploadFile = File(...),
    num_recommendations: int = 3
):
    """
    Upload an image and get fashion recommendations
    """
    global model, label_binarizers, metadata_df
    
    if model is None or label_binarizers is None or metadata_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        image = load_image(temp_file_path)
        
        predictions = predict_fashion_item(model, label_binarizers, image)
        
        input_metadata = {
            'gender': predictions['gender']['label'],
            'subCategory': predictions['subCategory']['label'],
            'baseColour': predictions['color']['label'],
            'season': predictions['season']['label'],
            'usage': predictions['usage']['label']
        }
        
        recommendations = get_fashion_recommendations(
            input_metadata, 
            metadata_df, 
            num_recommendations
        )
        
        recommendations = enhance_with_image_urls(recommendations)
        
        os.unlink(temp_file_path)
        
        return PredictionResponse(
            input_metadata=input_metadata,
            recommendations=recommendations,
            message="Recommendations generated successfully"
        )
        
    except Exception as e:
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/categories")
async def get_categories():
    """Get available categories from the dataset"""
    global metadata_df
    
    if metadata_df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    categories = {
        'subCategories': sorted(metadata_df['subCategory'].unique().tolist()),
        'colors': sorted(metadata_df['baseColour'].unique().tolist()),
        'seasons': sorted(metadata_df['season'].unique().tolist()),
        'usage': sorted(metadata_df['usage'].unique().tolist()),
        'genders': sorted(metadata_df['gender'].unique().tolist())
    }
    
    return categories

@app.get("/stats")
async def get_dataset_stats():
    """Get dataset statistics"""
    global metadata_df
    
    if metadata_df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    image_count = 0
    if os.path.exists(IMAGES_FOLDER):
        image_count = len([f for f in os.listdir(IMAGES_FOLDER) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    
    stats = {
        'total_items': len(metadata_df),
        'available_images': image_count,
        'categories_count': metadata_df['subCategory'].value_counts().to_dict(),
        'colors_count': metadata_df['baseColour'].value_counts().to_dict(),
        'gender_distribution': metadata_df['gender'].value_counts().to_dict(),
        'season_distribution': metadata_df['season'].value_counts().to_dict(),
        'usage_distribution': metadata_df['usage'].value_counts().to_dict(),
    }
    
    return stats

@app.get("/image/{image_id}")
async def get_image(image_id: str):
    """Get a specific image by ID"""
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    for ext in extensions:
        image_filename = f"{image_id}{ext}"
        image_path = os.path.join(IMAGES_FOLDER, image_filename)
        
        if os.path.exists(image_path):
            return FileResponse(image_path)
    
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/sample-items/{category}")
async def get_sample_items(category: str, limit: int = 10):
    """Get sample items from a specific category"""
    global metadata_df
    
    if metadata_df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    category_items = metadata_df[metadata_df['subCategory'] == category]
    
    if len(category_items) == 0:
        raise HTTPException(status_code=404, detail=f"No items found for category: {category}")
    
    sample_items = category_items.head(limit).to_dict('records')
    
    for item in sample_items:
        image_id = str(item.get('id', ''))
        if image_id:
            item['image_url'] = get_local_image_url(image_id)
    
    return {
        'category': category,
        'total_items': len(category_items),
        'sample_items': sample_items
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
#     import numpy as np
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import LabelBinarizer, RobustScaler
# from scipy import stats
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import Dense, Input, Flatten, BatchNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# def advanced_data_preparation(
#     image_data, 
#     df, 
#     sample_size=None, 
#     test_size=0.2, 
#     random_state=42, 
#     z_threshold=3,
#     verbose=True
# ):
#     """
#     Comprehensive data preparation function with:
#     - Sampling 
#     - Outlier removal
#     - Label binarization
#     - Stratified splitting
#     """
#     if sample_size is None:
#         sample_size = len(df)
    
#     image_data_sampled = image_data[:sample_size]
    
#     if verbose:
#         print("\nOriginal DataFrame columns:", df.columns)
#         for column in ['subCategory', 'gender', 'baseColour', 'season', 'usage']:
#             unique_values = df[column].nunique()
#             print(f"\nUnique values in {column}: {unique_values}")
#             print(f"Values: {sorted(df[column].unique())}")
    
    
#     label_mapping = {
#         'subCategory': 'subCategory',
#         'gender': 'gender',
#         'baseColour': 'color',
#         'season': 'season',
#         'usage': 'usage'
#     }
    
#     label_binarizers = {
#         output_name: LabelBinarizer()
#         for output_name in label_mapping.values()
#     }
    
#     labels_dict = {}
#     for input_name, output_name in label_mapping.items():
#         labels_dict[f'{output_name}_output'] = label_binarizers[output_name].fit_transform(
#             np.array(df[input_name].values[:sample_size])
#         )
    
#     def remove_outliers(data, labels_dict, z_threshold=3):
#         """Remove outliers using z-score method across multiple labels"""
#         total_samples = len(data)
#         keep_mask = np.ones(total_samples, dtype=bool)
        
#         if verbose:
#             print("\nInitial Shapes:")
#             print(f"Data shape: {data.shape}")
#             for name, labels in labels_dict.items():
#                 print(f"{name} shape: {labels.shape}")
        
#         for label_name, labels in labels_dict.items():
#             if verbose:
#                 print(f"\nProcessing {label_name}:")
#                 print(f"Initial samples: {np.sum(keep_mask)}")
            
#             label_indices = np.argmax(labels, axis=1)
#             z_scores = np.abs(stats.zscore(label_indices))
#             current_mask = z_scores < z_threshold
#             keep_mask = keep_mask & current_mask
            
#             if verbose:
#                 print(f"Samples after filtering: {np.sum(keep_mask)}")
        
#         clean_data = data[keep_mask]
#         clean_labels_dict = {
#             key: labels[keep_mask] for key, labels in labels_dict.items()
#         }
        
#         if verbose:
#             print("\nAfter removing outliers:")
#             print(f"Clean data shape: {clean_data.shape}")
#             for name, labels in clean_labels_dict.items():
#                 print(f"{name} shape: {labels.shape}")
        
#         return clean_data, clean_labels_dict
    
#     clean_data, clean_labels_dict = remove_outliers(
#         image_data_sampled, 
#         labels_dict, 
#         z_threshold=z_threshold
#     )
    
#     stratified_split = StratifiedShuffleSplit(
#         n_splits=1, 
#         test_size=test_size, 
#         random_state=random_state
#     )
    
#     primary_label = clean_labels_dict['subCategory_output']
#     primary_label_indices = np.argmax(primary_label, axis=1)
    
#     for train_index, test_index in stratified_split.split(clean_data, primary_label_indices):
#         trainX = clean_data[train_index]
#         testX = clean_data[test_index]
#         trainY_dict = {
#             key: labels[train_index] for key, labels in clean_labels_dict.items()
#         }
#         testY_dict = {
#             key: labels[test_index] for key, labels in clean_labels_dict.items()
#         }
    
#     num_classes_dict = {
#         key.replace('_output', ''): labels.shape[1] 
#         for key, labels in trainY_dict.items()
#     }
    
#     if verbose:
#         print("\nFinal training shapes:")
#         print(f"trainX shape: {trainX.shape}")
#         for name, labels in trainY_dict.items():
#             print(f"{name} shape: {labels.shape}")
#         print("\nNumber of Classes:")
#         for name, num_classes in num_classes_dict.items():
#             print(f"{name}: {num_classes} classes")
    
#     return {
#         'trainX': trainX,
#         'trainY_dict': trainY_dict,
#         'testX': testX,
#         'testY_dict': testY_dict,
#         'num_classes_dict': num_classes_dict,
#         'label_binarizers': label_binarizers
#     }

