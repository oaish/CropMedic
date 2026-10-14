from pathlib import Path
from predict import TrainedDiseaseClassifier
from config import TEST_DIR


def test_all():
    classifier = TrainedDiseaseClassifier()

    print("\n" + "=" * 65)
    print("VERIFYING MODEL PREDICTIONS ON UNSEEN TEST SAMPLES FROM ALL 8 CLASSES")
    print("=" * 65)

    for cls_dir in sorted(TEST_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue

        images = list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.jpg"))
        if not images:
            continue

        sample_img = images[0]
        preds = classifier.predict_file(sample_img)

        top_pred = preds[0]
        runner_up = preds[1]

        print(f"Ground Truth : {cls_dir.name}")
        print(f"Sample File  : {sample_img.name}")
        print(f"Prediction   : {top_pred['disease']} ({top_pred['confidence'] * 100:.2f}%)")
        print(f"Runner-up    : {runner_up['disease']} ({runner_up['confidence'] * 100:.2f}%)")
        print("-" * 65)


if __name__ == "__main__":
    test_all()
