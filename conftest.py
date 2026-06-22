import sys
from pathlib import Path

# Adiciona src/ ao path para que os testes encontrem os módulos
sys.path.insert(0, str(Path(__file__).parent / "src"))