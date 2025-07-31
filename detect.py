import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('train/exp4/weights/best.pt') # select your model.pt path
    model.predict(source='D:\\yintongg\\pcb',
                  conf=0.25,
                  project='runs/detect',
                  name='exp',
                  save=True,
                 #visualize=True # visualize model features maps
                  )