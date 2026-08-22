"""Put the component root on `sys.path` so `import image_processor.pipeline` resolves from anywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
