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
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']

class_explanations = {
    "Glioma": "Gliomas are tumors that arise from glial cells in the brain or spine. They can be aggressive and may require surgery, radiation, or chemotherapy.",
    "Meningioma": "Meningiomas are typically benign tumors that form on membranes covering the brain and spinal cord. They grow slowly and are often treatable with surgery.",
    "No Tumor": "No tumor was detected in the image. This suggests that the MRI scan appears normal based on the model's analysis.",
    "Pituitary Tumor": "Pituitary tumors form in the pituitary gland at the base of the brain. Most are noncancerous and may affect hormone production."
}

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
    image = np.array(image)  # normalize
    if image.shape[-1] == 4:  # Remove alpha channel if present
        image = image[..., :3]
    image = np.expand_dims(image, axis=0) 

    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0])) * 100

    return {"prediction": predicted_class, "confidence": confidence,
            "explanation": class_explanations.get(predicted_class, "No explanation available.")}
