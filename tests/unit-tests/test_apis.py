from fastapi.testclient import TestClient
from src.app.app import app
from tests.helpers import *

client = TestClient(app)

def test_predict_torch():
    response = predict_test(client, "/predict/torch_model/")
    assert response["status_code"] == 200
