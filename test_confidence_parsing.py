import pytest

from app import extract_confidence


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Confidence Score: 87%", 87.0),
        ("confidence: 0.87", 87.0),
        ("Confidence is 87 percent", 87.0),
        ("87%", 87.0),
        ("Confidence=100%", 100.0),
        ("Confidence Score: 45.5%", 45.5),
        ("Confidence: .92", 92.0),
        ("No confidence here", None),
        ("Confidence: 0.0", 0.0),
        ("confidence 1.00", 100.0),
    ],
)
def test_extract_confidence_various_formats(text, expected):
    result = extract_confidence(text)
    if expected is None:
        assert result is None
    else:
        assert pytest.approx(result, rel=1e-3) == expected
