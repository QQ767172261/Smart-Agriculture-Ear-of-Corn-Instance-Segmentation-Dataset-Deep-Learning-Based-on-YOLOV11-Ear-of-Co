智慧农业-玉米穗实例分割数据集 深度学习基于YOLOV11玉米穗实例分割系统  超高分辨率玉米穗实例分割数据集，3689张，提供yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b99b5f6a40c1448d90ca5450ecc99f76.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1cccd4fc2ab54413952012eace0ab0cb.png)
数据集统计信息
====================
图像尺寸:4032*3024
类别数量: 1 类
训练集图像:3127; 验证集图像:373; 测试集图像:189
总图像数: 3689
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d717dba96a4c415c8c75b4dbaefda42d.png)

类别分布:
类别名称 | 包含该类别的图像数 | 总标注数
----------------------------------------
corn | 3689 | 3690


模型代码：模型训练使用yolov11n-seg训练，80个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f3e2c530a5f849fdb5203123d1142540.png)

qt界面：运行界面采用pyqt5编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像（
4.（支持windows和linux系统），


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0aa92e447b03495aa616e878f35139f0.png)


## 1. 环境安装

```bash
pip install ultralytics opencv-python PyQt5 numpy pillow
```

## 2. 数据集配置文件 `corn.yaml`![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ff74033669b844d29b51326cf8f65cb1.png)


```yaml
path: ./corn_dataset
train: images/train
val: images/val
test: images/test
names:
  0: corn
```

## 3. 训练脚本 `train.py`![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4db12318851743bbac90ef3d759793f6.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/471256b7e720460987745bbf962dedd7.png)


```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.train(
        data="corn.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        patience=10,
        save=True,
        project="runs/train",
        name="corn_detect"
    )
```

训练完成后，把最佳权重复制到主程序目录：

```bash
runs/train/corn_detect/weights/best.pt
```

## 4. 预测脚本 `predict.py`![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d15556b586be4106b3705d9e0ed080ae.png)


```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test_corn.jpg",
    conf=0.5,
    save=True
)

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        print(f"类别: {cls_name}, 置信度: {conf:.3f}, 坐标: {xyxy}")
```

## 5. PyQt5 可视化检测界面 `main.py`

```python
import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO


class DetectThread(QThread):
    result_signal = pyqtSignal(dict)

    def __init__(self, model, source, mode="img"):
        super().__init__()
        self.model = model
        self.source = source
        self.mode = mode
        self.running = True

    def run(self):
        if self.mode == "img":
            self.detect_image(self.source)
        elif self.mode == "video":
            self.detect_video(self.source)
        elif self.mode == "cam":
            self.detect_video(0)

    def detect_image(self, path):
        result = self.model.predict(path, conf=0.5)[0]
        self.emit_result(result)

    def detect_video(self, source):
        cap = cv2.VideoCapture(source)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = self.model.predict(frame, conf=0.5)[0]
            self.emit_result(result)

        cap.release()

    def emit_result(self, result):
        plot_img = result.plot()
        boxes = []

        for box in result.boxes:
            cls_name = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            xyxy = [int(x) for x in box.xyxy[0]]

            boxes.append({
                "cls": cls_name,
                "conf": conf,
                "xyxy": xyxy
            })

        self.result_signal.emit({
            "img": plot_img,
            "boxes": boxes
        })

    def stop(self):
        self.running = False


class CornDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的玉米检测系统")
        self.resize(1280, 900)
        self.model = YOLO("best.pt")
        self.thread = None
        self.current_result = None
        self.init_ui()

    def init_ui(self):
        self.label_image = QLabel(self)
        self.label_image.setGeometry(20, 80, 640, 500)
        self.label_image.setStyleSheet("border: 1px solid gray;")
        self.label_image.setAlignment(Qt.AlignCenter)

        self.btn_image = QPushButton("图片检测", self)
        self.btn_image.setGeometry(700, 80, 160, 40)
        self.btn_image.clicked.connect(self.open_image)

        self.btn_video = QPushButton("视频检测", self)
        self.btn_video.setGeometry(700, 130, 160, 40)
        self.btn_video.clicked.connect(self.open_video)

        self.btn_camera = QPushButton("摄像头实时", self)
        self.btn_camera.setGeometry(700, 180, 160, 40)
        self.btn_camera.clicked.connect(self.open_camera)

        self.label_conf = QLabel("置信度：--", self)
        self.label_conf.setGeometry(700, 250, 200, 30)

        self.label_pos = QLabel("目标位置：--", self)
        self.label_pos.setGeometry(700, 300, 240, 100)

        self.table = QTableWidget(self)
        self.table.setGeometry(20, 600, 640, 240)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号", "类别", "置信度", "xmin", "ymin", "xmax", "ymax"])

        self.btn_save = QPushButton("保存结果", self)
        self.btn_save.setGeometry(700, 480, 130, 40)
        self.btn_save.clicked.connect(self.save_result)

        self.btn_exit = QPushButton("退出", self)
        self.btn_exit.setGeometry(850, 480, 130, 40)
        self.btn_exit.clicked.connect(self.close)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png)"
        )

        if not path:
            return

        self.stop_thread()
        self.thread = DetectThread(self.model, path, mode="img")
        self.thread.result_signal.connect(self.show_result)
        self.thread.start()

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频 (*.mp4 *.avi)"
        )

        if not path:
            return

        self.stop_thread()
        self.thread = DetectThread(self.model, path, mode="video")
        self.thread.result_signal.connect(self.show_result)
        self.thread.start()

    def open_camera(self):
        self.stop_thread()
        self.thread = DetectThread(self.model, 0, mode="cam")
        self.thread.result_signal.connect(self.show_result)
        self.thread.start()

    def show_result(self, data):
        self.current_result = data
        image = data["img"]
        boxes = data["boxes"]

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        q_image = QImage(
            rgb_image.data, width, height,
            width * channels, QImage.Format_RGB888
        )
        self.label_image.setPixmap(
            QPixmap.fromImage(q_image).scaled(
                self.label_image.size(), Qt.KeepAspectRatio
            )
        )

        self.table.setRowCount(len(boxes))

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box["xyxy"]

            self.table.setItem(index, 0, QTableWidgetItem(str(index + 1)))
            self.table.setItem(index, 1, QTableWidgetItem(box["cls"]))
            self.table.setItem(index, 2, QTableWidgetItem(f"{box['conf']:.2%}"))
            self.table.setItem(index, 3, QTableWidgetItem(str(x1)))
            self.table.setItem(index, 4, QTableWidgetItem(str(y1)))
            self.table.setItem(index, 5, QTableWidgetItem(str(x2)))
            self.table.setItem(index, 6, QTableWidgetItem(str(y2)))

        if boxes:
            self.label_conf.setText(f"置信度 {boxes[0]['conf']:.2%}")
            x1, y1, x2, y2 = boxes[0]["xyxy"]
            self.label_pos.setText(
                f"xmin: {x1}\nymin: {y1}\nxmax: {x2}\nymax: {y2}"
            )

    def save_result(self):
        if self.current_result is None:
            QMessageBox.warning(self, "提示", "暂无检测结果！")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "corn_result.jpg", "图片 (*.jpg)"
        )

        if path:
            cv2.imwrite(path, self.current_result["img"])
            QMessageBox.information(self, "成功", "结果已保存！")

    def stop_thread(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None

    def closeEvent(self, event):
        self.stop_thread()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CornDetectWindow()
    window.show()
    sys.exit(app.exec_())
```

## 6. 使用方式

```bash
# 训练模型
python train.py

# 单图推理
python predict.py

# 启动可视化界面
python main.py
```

## 7. 功能说明

- 支持图片、视频、摄像头实时检测
- 自动识别玉米目标 `corn`
- 输出检测框、类别、置信度、坐标
- PyQt5 界面显示检测结果与目标信息表
- 可保存检测结果图片


