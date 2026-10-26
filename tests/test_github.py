import httpx

from hunkrisk.github import API_URL, GitHub, last_page

LINK = (
    '<https://api.github.com/repositories/1/commits?per_page=100&page=2>; rel="next", '
    '<https://api.github.com/repositories/1/commits?per_page=100&page=57>; rel="last"'
)


def test_last_page_reads_rel_last():
    assert last_page(LINK) == 57


def test_last_page_without_pagination_is_one():
    assert last_page(None) == 1


def test_get_passes_path_as_a_query_parameter_and_caches(tmp_path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[{"sha": "abc"}], headers={"link": LINK})

    gh = GitHub(tmp_path, token="test")
    gh.client = httpx.Client(base_url=API_URL, transport=httpx.MockTransport(handler))

    first = gh.get("/repos/a/b/commits", path="src/auth", per_page=100)
    second = gh.get("/repos/a/b/commits", path="src/auth", per_page=100)

    assert first == second == ([{"sha": "abc"}], LINK)
    assert len(requests) == 1
    assert requests[0].url.params["path"] == "src/auth"
