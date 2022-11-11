from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

MODEL = tf.keras.models.load_model("Tensorflow/Computer Vision/Flowers_Classification/Flowers_model.h5")

CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file:UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image,[224,224])
    image = image/255.0
    image = tf.expand_dims(image,axis=0)
    predictions = MODEL.predict(image)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        'class': predicted_class,
        'confidence' : confidence
    }

if __name__ =="__main__":
    uvicorn.run(app, host='localhost',port=8000)

