
import ultralytics
from ultralytics import YOLO

if __name__ == "__main__":
  path = "C:/L2M/train_delineation/"
  model_path = path + "best.pt"
  model = YOLO(model_path)  # load a custom model

  # source_image = path + "datasets/train-01/images/04d54604-frame0132.jpg"
  dataset = 'C:/L2M/TrainData/Delineation/'
  source_image = dataset + "dataset/val/images/"

  source_image = "C:/L2M/TrainData/Process/T20231013/T20231013115757/frames/"
  source_image = "C:/L2M/TrainData/Delineation/dataset/val/images/"

  predict_path = dataset + "/predicts/"

  result = model.predict(
    source=source_image,
    device=0,
    save=True,
    conf=0.5,
    save_crop=True,
    save_txt=True,
    save_conf=True,
    show_labels=True,
    show_conf=True,
    name=f"{predict_path}coach",
    half=True,
    stream=False,
  )

  #print(result)
