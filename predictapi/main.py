from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add the middleware to your app
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",            # Allows specific origins  allow_origins=origins
    allow_credentials=True,
    allow_methods=["*"],              # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],              # Allows all headers
)

CLASS_NAMES = ['Tomato__Bacterial_spot',
 'Tomato__Early_blight',
 'Tomato__Late_blight',
 'Tomato__Leaf_Mold',
 'Tomato__Septoria_leaf_spot',
 'Tomato__Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato__healthy']
MODEL_PATH = '../saved_models/1.keras'

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        #model = tf.keras.models.load_model(model_path, compile=False)
        print(model)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

MODEL = load_model(MODEL_PATH)
if not MODEL:
    print("Error while loading Model")

@app.get("/ping")
async def ping():
    return "The Service is alive"

def file_to_image_numpy(data) -> np.ndarray:
    np_image = np.array(Image.open(BytesIO(data)))
    return np_image

@app.post("/predict")
async def predict( file: UploadFile = File(...)):

    try:
        img_np = file_to_image_numpy(await file.read())

        ## image is [256, 256, 3]
        ## predict needs batch [batch_no, 256, 256, 3]
        img_batch = np.expand_dims(img_np, axis=0)
        predictions = MODEL.predict(img_batch)
        print(predictions)
        print("argmax", np.argmax(predictions[0]), "max",  np.max(predictions[0]))

        predicted_label = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        print({ "class": predicted_label, "confidence": confidence })
        return {
            "class": predicted_label,
            "confidence": float(confidence),
            "prediction": predictions.tolist()[0]
        }

    except Exception as e:
        print(f"Error while predicting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8010)