from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pydantic import BaseModel
import pytesseract

app = FastAPI()

class FileInput(BaseModel):
    file_path: str

class DocumentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DocumentClassifier, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the pre-trained model
model = DocumentClassifier(num_classes=4)
model.load_state_dict(torch.load("fine_tuned_model.pth"))  # Specify the correct path
model.eval()

document_types = {0: "Pan Card", 1: "Aadhar Card", 2: "Passport", 3: "Voter ID"}

def perform_ocr(image_path):
    # Open the image using PIL
    img = Image.open(image_path)

    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(img)

    return text

def check_if_front_or_back(doc, image_path):
    extracted_text = perform_ocr(image_path)
    # print(extracted_text)
    if doc == 'Aadhar Card':
        if 'Address' in extracted_text:
            return "Back"
        else:
            return "Front"   

    if doc == "Pan Card":
        if 'Permanent Account Number' in extracted_text:
            return "Front"  
        else:
            return "Back"
    
    if doc == "Passport":
        if 'Date of Birth:' in extracted_text:
            return "Front"
        else:
            return "Back"

    if doc == "Voters ID":
        if "Name" in extracted_text:
            return "Front"
        else:
            return "Back"

def predict_single_document(file_path):
    try:
        # Load image
        image = Image.open(file_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        image = transform(image)
        image = image.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(image)

        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()

        # Map predicted class to document type
        predicted_document = document_types[predicted_class]

        front_or_back = check_if_front_or_back(predicted_document, file_path)

        return {"document_type": predicted_document, "view": front_or_back}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing {file_path}: {e}")

@app.post("/classify")
async def classify_document(file_input: FileInput):
    result = predict_single_document(file_input.file_path)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
