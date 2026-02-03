# DefectDetector Service

A defect detection microservice for fruit analysis systems. This service receives cropped fruit images and returns defect segmentation results using YOLO-Seg model inference.

## Overview

The DefectDetector is a passive microservice that:
- Receives cropped fruit images from the Brain orchestrator
- Detects defects using YOLO-Seg model
- Returns defect results with segmentation polygons

**Note:** This service never initiates requests—it only responds to requests from the Brain service.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect-defects` | POST | Detect defects in a cropped fruit image |
| `/health` | GET | Health check endpoint |
| `/` | GET | Service information |

### POST `/detect-defects`

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `image` | file | Binary crop of one fruit (JPEG/PNG) |
| `image_id` | string | ID of the original full image |
| `fruit_id` | string | ID of the fruit assigned by Brain |

**Response:**
```json
{
  "image_id": "string",
  "fruit_id": "string",
  "defects": [
    {
      "type": "defect",
      "confidence": 0.93,
      "segmentation": {
        "polygon": [[x1, y1], [x2, y2], ...]
      }
    }
  ]
}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_PORT` | `8400` | Port to run the service on |
| `SERVICE_HOST` | `0.0.0.0` | Host to bind to |
| `SERVICE_NAME` | `DefectDetector` | Service name |
| `DEBUG` | `false` | Enable debug mode |
| `MODEL_PATH` | `models/defect_seg.pt` | Path to model weights |
| `CONFIDENCE_THRESHOLD` | `0.3` | Detection confidence threshold |
| `INFERENCE_DEVICE` | `cpu` | Inference device (`cpu`, `cuda`, `cuda:0`) |
| `IMAGE_SIZE` | `640` | YOLO inference image size |

---

## Docker

### Building the Image

```bash
docker build -t defect-detector:latest .
```

### Running the Container

Basic run command:

```bash
docker run --rm -p 8400:8400 defect-detector:latest
```

With custom configuration:

```bash
docker run --rm -p 8400:8400 \
  -e SERVICE_PORT=8400 \
  -e CONFIDENCE_THRESHOLD=0.5 \
  -e INFERENCE_DEVICE=cpu \
  defect-detector:latest
```

### Using Docker Compose

Start the service:

```bash
docker compose up -d
```

Stop the service:

```bash
docker compose down
```

View logs:

```bash
docker compose logs -f defect-detector
```

### Health Check

Verify the service is running:

```bash
curl http://localhost:8400/health
```

Expected response:
```json
{"status": "healthy", "service": "DefectDetector"}
```

### GPU Support (Optional)

To run with CUDA GPU support, ensure you have the NVIDIA Container Toolkit installed, then:

```bash
docker run --rm --gpus all -p 8400:8400 \
  -e INFERENCE_DEVICE=cuda \
  defect-detector:latest
```

---

## Local Development

### Prerequisites

- Python 3.10+
- Model file at `models/defect_seg.pt`

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn app.main:app --host 0.0.0.0 --port 8400 --reload
```

Or use the provided script:

```bash
./uvicorn.sh
```

### Testing the Endpoint

```bash
curl -X POST http://localhost:8400/detect-defects \
  -F "image=@test_fruit.jpg" \
  -F "image_id=img_001" \
  -F "fruit_id=fruit_001"
```

## Project Structure

```
defectservice/
├── app/
│   ├── __init__.py
│   ├── api.py          # API endpoints
│   ├── config.py       # Configuration settings
│   ├── infer.py        # Model inference logic
│   ├── logging_config.py
│   ├── main.py         # FastAPI application
│   ├── models.py       # Pydantic models
│   └── utils.py        # Utility functions
├── models/
│   └── defect_seg.pt   # YOLO-Seg model weights
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## License

Proprietary - Part of the Fruit Analysis System.

