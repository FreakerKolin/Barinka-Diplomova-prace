import os
from src.train import train_model
from src.apply import apply_model

#spouštěcí blok
if __name__ == "__main__":
    model_path = "models/cnn_model.pth"

    # trénink, pokud model ještě neexistuje
    if not os.path.exists(model_path):
        train_model(
            input_dir="data/input_wavs",
            target_dir="data/target_wavs",
            save_path=model_path,
            epochs=100
        )
    else:
        print("Model už existuje, netrénuju znovu.")

    # testovací soubory
    test_dir = "data/test_wavs"
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".wav")]

    for f in test_files:
        input_file = os.path.join(test_dir, f)
        output_file = os.path.join(test_dir, f.replace(".wav", "_processed.wav"))
        apply_model(model_path, input_file, output_file)