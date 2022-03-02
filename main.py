import cv2
import detector
from num2code import *
from torchvision.transforms import ToTensor


class Detector:
    def __init__(self, model, **kwargs):
        self._size = (640, 480)
        self.model = model(**kwargs)
        self.model.eval()
        self.tensor = ToTensor()

    def __call__(self, path, output, show=False):
        num = [0] * 49
        img = cv2.resize(cv2.imread(path), self._size)
        bls = self.model([self.tensor(img)])[0][0]
        bls = [bls["boxes"], bls["labels"], bls["scores"]]
        boxes, labels, scores = list(map(lambda x: x.detach().numpy().tolist(), bls))
        for i in range(len(boxes)):
            box, label, score = boxes[i], labels[i], scores[i]
            if score < 0.4:
                continue
            if show:
                text = num2code[label]
                score = str(int(score * 100) / 100)[:4]
                text += "(" + score + ")"
                font = cv2.FONT_HERSHEY_SIMPLEX
                x1, y1, x2, y2 = list(map(int, box))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
                cv2.putText(img, text, (x1, y1), font, 0.4, (0, 0, 255))
            num[label] += 1
        if show:
            cv2.namedWindow(show)
            cv2.imshow(show, img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        out = ["START"]
        for i in range(49):
            if num[i]:
                text = "Goal_ID=" + num2code[i] + ";Num=" + str(num[i])
                out.append(text)
        out.append("END")
        with open(output, "w") as txt:
            txt.write("\n".join(out))


ai = Detector(
    detector.CascadeRCNN,
    number=49, iou1=0.5, iou2=0.6, iou3=0.7,
    rpn="parameters/rpn.pkl",
    net0="parameters/net.pkl",
    cascade="parameters/cascade.pkl",
    tail="parameters/tail.pkl"
)

if __name__ == "__main__":
    while True:
        image_path = input("Image path:")
        output_file = input("Output file name:")
        while True:
            window = input("Show result?(y/n)")
            if window == "y":
                window = input("Window name:")
                break
            if window == "n":
                window = False
                break
        ai(image_path, output_file, window)
