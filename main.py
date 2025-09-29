import os
from src.train import train_model
from src.apply import apply_model

if __name__ == "__main__":
    model_path = "models/cnn_model.pth"

    # jen pokud model neexistuje
    if not os.path.exists(model_path):
        train_model(
            input_dir="data/input_wavs",
            target_dir="data/target_wavs",
            save_path=model_path,
            epochs=500
        )
    else:
        print("Model už existuje, netrénuju znovu.")

    # aplikace na testovací zvuk
    test_files = [f for f in os.listdir("data/test_wavs") if f.endswith(".wav")]
    for f in test_files:
        input_file = os.path.join("data/test_wavs", f)
        output_file = os.path.join("data/test_wavs", f.replace(".wav", "_processed.wav"))
        apply_model(model_path, input_file, output_file)