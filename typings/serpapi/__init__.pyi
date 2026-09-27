class SerpApiClient:
    def __init__(
        self,
        params_dict: dict[str, str],
        engine: str | None = None,
        timeout: int = 60000,
    ) -> None: ...

    def get_dict(self) -> dict[str, object]: ...
