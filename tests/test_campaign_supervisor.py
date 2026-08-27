from scripts.campaign_supervisor import _dotenv_value


def test_dotenv_value_strips_matching_quotes() -> None:
    assert _dotenv_value('"https://example.invalid/mlflow"') == (
        "https://example.invalid/mlflow"
    )
    assert _dotenv_value("'secret-value'") == "secret-value"


def test_dotenv_value_preserves_unquoted_and_unmatched_values() -> None:
    assert _dotenv_value("  raw-value  ") == "raw-value"
    assert _dotenv_value('"unmatched') == '"unmatched'
