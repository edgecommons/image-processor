"""Put the component root on `sys.path` so `import image_processor.pipeline` resolves from anywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# WP7 -- the tier-2 real-model suite writes its goldens from a real run rather than by hand. The
# switch is registered here because pytest only reads `pytest_addoption` from the root conftest,
# so `--update-goldens` is valid whether the suite is run on its own or with everything else.
def pytest_addoption(parser):
    """Register the golden-writing switch (WP7).

    Args:
        parser: pytest's option parser.
    """
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="write tests/goldens/ from this run instead of asserting against it",
    )
