import os
from ultralytics import YOLO


def main():
    # Load a model
    model = YOLO("yolov8n.pt")

    # Train the model
    train_results = model.train(
        data="./coco8.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        project=f"{os.path.dirname(os.path.abspath(__file__))}/yolo",
        name="runs",
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("./test/Petland_Florida_Cavalier_King_Charles_Spaniel_puppy.jpg")
    results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model


if __name__ == "__main__":
    main()
