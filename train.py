from ultralytics import YOLO
from datetime import datetime

if __name__ == '__main__':
  # Load a model
  model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)

  # Train the model
  dataset = 'C:/L2M/TrainData/Delineation/dataset/data.yaml'

  start_time = datetime.now()

  results = model.train(data=dataset, epochs=3, imgsz=640)

  end_time = datetime.now()
  print("Time taken:", str(end_time - start_time))


# results = model.train(
#    data='custom_data.yaml',
#    imgsz=640,
#    epochs=10,
#    batch=8,
#    name='yolov8n_custom'