from main import main
import pytest


def test_main():
    """Test d'exécution de la fonction main"""
    with pytest.raises(SystemExit):  # car argparse peut lever une sortie
        main()
