from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
from rapidocr_onnxruntime import RapidOCR
import requests
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


vision_client = ImageAnalysisClient(
    endpoint=AZURE_VISION_ENDPOINT, 
    credential=AzureKeyCredential(AZURE_VISION_KEY)
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

# ─── Helpers ──────────────────────────────────────────────────────────────────

def analyze_led_circuit_from_bytes(image_bytes: bytes) -> dict:
    """Detect the dominant LED colour in an image using HSV analysis."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Impossibile decodificare l'immagine")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    blurred_v = cv2.GaussianBlur(v_channel, (15, 15), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred_v)
    x, y = max_loc

    if max_val < 220:
        return {"category": "led", "value": "off", "confidence": 0.0}

    padding = 40
    h, w = img.shape[:2]
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(w, x + padding), min(h, y + padding)

    hsv_crop = hsv[y1:y2, x1:x2]
    color_mask = cv2.inRange(hsv_crop, (0, 100, 150), (180, 255, 255))
    valid_pixels = hsv_crop[color_mask > 0]

    if len(valid_pixels) == 0:
        return {"category": "led", "value": "indeterminato", "confidence": 0.0}

    color_ranges = {
        "red":    [(0, 10), (165, 180)],
        "yellow": [(15, 35)],
        "green":  [(45, 85)],
        "blue":   [(100, 130)],
        "purple": [(135, 160)],
    }

    hues = valid_pixels[:, 0]
    votes: dict[str, int] = {}
    for color, ranges in color_ranges.items():
        count = sum(np.sum((hues >= low) & (hues <= high)) for low, high in ranges)
        votes[color] = int(count)

    winner = max(votes, key=votes.get)
    confidence = votes[winner] / len(valid_pixels)

    if confidence < 0.20:
        return {"category": "led", "value": "undefined", "confidence": round(float(confidence), 4)}

    return {"category": "led", "value": winner, "confidence": round(float(confidence), 4)}

def get_local_background_category(img, bbox) -> str:
    """Classify the local background around a text bounding-box as 'document' or 'box'.
    bbox: array di 4 punti [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (formato RapidOCR)
    """
    pts = np.array(bbox, dtype=np.int32)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    h, w, _ = img.shape
    pad = 10
    roi = img[max(0, y_min - pad):min(h, y_max + pad),
              max(0, x_min - pad):min(w, x_max + pad)]

    if roi.size == 0:
        return "document"

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    median_hsv = np.median(hsv_roi.reshape(-1, 3), axis=0)
    h_val, s_val, v_val = median_hsv

    if v_val > 170 and s_val < 60:
        return "document"
    elif 5 <= h_val <= 35 and s_val > 40:
        return "box"
    else:
        return "document"

def analyze_text_from_bytes(image_bytes: bytes) -> dict:
    """Usa Azure Vision per l'OCR e OpenCV per l'analisi del background."""
    
    try:
        # Chiamata ad Azure
        result = vision_client.analyze(
            image_data=image_bytes,
            visual_features=[VisualFeatures.READ]
        )
    except Exception as e:
        raise ValueError(f"Errore chiamata Azure: {str(e)}")

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Impossibile decodificare l'immagine")

    final_output = []

    if result.read and result.read.blocks:
        for line in result.read.blocks[0].lines:
            text = line.text
            
            # CORREZIONE QUI: Azure restituisce oggetti Point(x, y)
            p = line.bounding_polygon
            box = [[p[0].x, p[0].y], [p[1].x, p[1].y], [p[2].x, p[2].y], [p[3].x, p[3].y]]
            
            score = 0.99 

            category = get_local_background_category(img, box)
            final_output.append({
                "category": category,
                "value": text,
                "confidence": score
            })

    if not final_output:
        return {"category": "unknown", "value": "", "confidence": 0.0}

    return max(final_output, key=lambda x: x["confidence"])

# ─── Shared validation ────────────────────────────────────────────────────────

def _validate_request(file: UploadFile, x_api_key: str | None) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
def _validate_command_request(text: str | None, x_api_key: str | None) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not text:
        raise HTTPException(status_code=400, detail="None command string.")

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)

API_KEY = os.environ["API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

AZURE_VISION_KEY = os.environ.get("AZURE_VISION_KEY")
AZURE_VISION_ENDPOINT = os.environ.get("AZURE_VISION_ENDPOINT")

@app.get("/")
def root():
    return {"message": "Computer Vision API is running."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/led")
async def predict_led(
    file: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    _validate_request(file, x_api_key)
    try:
        image_bytes = await file.read()
        led_results = analyze_led_circuit_from_bytes(image_bytes)
        return led_results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/text")
async def predict_text(
    file: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    _validate_request(file, x_api_key)
    try:
        image_bytes = await file.read()
        text_results = analyze_text_from_bytes(image_bytes)
        return text_results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/command")
async def predict_sentiment(
    request: Request,
    x_api_key: str = Header(None),
):

    body = await request.json()
    text = body.get("text", "").strip()

    _validate_command_request(text, x_api_key)


    request_text = "Riceverai una frase, voglio che tu capisca se si tratti di un comando di tipo HELP,STOP,REPEAT,UNKNOWN. Devi restituire solamente una di queste 4 parole e nient'altro la frase è la seguente: " + text

    try:
        openai_response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Sei un assistente che riceve una frase in linguaggio naturale e che deve capire di che comando si tratta"},
                    {"role": "user", "content": request_text},
                ],
            },
        )
        openai_response.raise_for_status()
        result = openai_response.json()
        content = result["choices"][0]["message"]["content"]
        return JSONResponse(content={"command": content})

    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Errore OpenAI: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
