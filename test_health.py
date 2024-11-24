import requests

def test_health_check():
    url = "http://localhost:5000/health"
    response = requests.get(url)
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}
