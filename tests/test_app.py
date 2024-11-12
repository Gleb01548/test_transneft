import yaml
import requests


url = yaml.load(open("./conf.yaml"), Loader=yaml.FullLoader)["url"]

def test_200():
    response = requests.get(f"{url}/health_check")
    assert response.status_code == 200


def test_422():
    response = requests.post(f"{url}/search", json={"лол": "кек"})
    assert response.status_code == 422

    response = requests.post(
        f"{url}/search", json={"query": None, "search_engine": "comb", "k": 10}
    )
    assert response.status_code == 422

    response = requests.post(
        f"{url}/search",
        json={
            "query": "Газпром преобразован в акционерное общество",
            "search_engine": None,
            "k": 10,
        },
    )
    assert response.status_code == 422

    response = requests.post(
        f"{url}/search",
        json={
            "query": "Газпром преобразован в акционерное общество",
            "search_engine": "comb",
            "k": None,
        },
    )
    assert response.status_code == 422


def test_query():

    response = requests.post(
        f"{url}/search",
        json={
            "query": "Газпром преобразован в акционерное общество",
            "search_engine": "comb",
            "k": 10,
        },
    )
    assert list(response.json().keys()) == ["status", "time_search", "search_result"]
    assert len(response.json()["search_result"]) == 10
    assert set([i["method"] for i in response.json()["search_result"]]) == set(
        ["vector", "bm25"]
    )

    response = requests.post(
        f"{url}/search",
        json={
            "query": "Газпром преобразован в акционерное общество",
            "search_engine": "vector",
            "k": 5,
        },
    )
    assert len(response.json()["search_result"]) == 5
    assert set([i["method"] for i in response.json()["search_result"]]) == set(
        ["vector"]
    )

    response = requests.post(
        f"{url}/search",
        json={
            "query": "Газпром преобразован в акционерное общество",
            "search_engine": "bm25",
            "k": 1,
        },
    )
    assert len(response.json()["search_result"]) == 1
    assert response.json()["search_result"][0]["method"] == "bm25"
