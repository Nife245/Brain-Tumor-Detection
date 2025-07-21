from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Load modelnotebook0d9f1c7422
model = tf.keras.models.load_model("model/brain_tumor_model.h5")
print("Model loaded successfully.")

# Define class names (adjust based on your model)
class_names = np.load("static/Tumor_classes.npy", allow_pickle=True).tolist()

print("Class names loaded successfully.")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# If using UI
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((224, 224))  # adjust to your input size
    image = np.array(image) / 255.0  # normalize
    if image.shape[-1] == 4:  # Remove alpha channel if present
        image = image[..., :3]
    image = np.expand_dims(image, axis=0) 

    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0])) * 100

    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

    return {"prediction": predicted_class, "confidence": confidence}
