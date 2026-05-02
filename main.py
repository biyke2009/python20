from fastapi import FastAPI, UploadFile, File, HTTPException
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image


class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CheckImage()
model.load_state_dict(torch.load("model_MNIST.pth", map_location=device))
model.to(device)
model.eval()
@app.post("/predict/")
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=404, detail="Файл пуст")

        img = Image.open(io.BytesIO(image_data))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            y_pred = y_pred.argmax(dim=1).item()

        return {"prediction": y_pred}

    except Exception as e:
        raise HTTPException(status_code=504, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8007)


