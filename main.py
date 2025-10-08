import os
from src.train import train_model
from src.apply import apply_model 

def get_latest_model(model_dir="models"):
    """
    Vrátí cestu k nejnovějšímu uloženému modelu (.pth) ve složce model_dir.
    Pokud složka nebo žádný model neexistuje, vrací None.

    """
    if not os.path.exists(model_dir):
        return None
    models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not models:
        return None
    # seřadí podle názvu (timestamp v názvu) a vezme poslední
    models.sort()
    return os.path.join(model_dir, models[-1])

if __name__ == "__main__":  # zjištění cesty k nejnovějšímu modelu
    model_path = get_latest_model()

    # trénink, pokud model ještě neexistuje
    if model_path is None:
        train_model(
            input_dir="data/input_wavs",
            target_dir="data/target_wavs",
            save_dir="models",
            epochs=100
        )
        model_path = get_latest_model()  # načte nově vytvořený model
    else:
        print(f"Model již existuje, použiji {model_path}")

    # testovací wav soubory
    test_dir = "data/test_wavs"
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".wav")]

    # Smyčka přes všechny testovací soubory, Aplikace natrénovaného modelu a uložení výstupů
    for f in test_files:
        input_file = os.path.join(test_dir, f)
        output_file = os.path.join(test_dir, f.replace(".wav", "_processed.wav"))
        apply_model(model_path, input_file, output_file)