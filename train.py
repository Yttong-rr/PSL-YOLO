import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/PSL-YOLO.yaml')
    model.load('D:\\liulanqidown\\11pt\\yolo11n.pt') # loading pretrain weights
    model.train(data='./neu.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=False, # close amp
                # fraction=0.2,
                project='train',
                name='exp',
                )