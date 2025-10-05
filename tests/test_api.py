import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient


def test_health_endpoint():
    from src.api import  app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
