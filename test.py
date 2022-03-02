import cv2
import data
import detector
from num2code import *
from torchvision.transforms import ToTensor


class Tester:
    def __init__(self, model, **kwargs):
        self.tensor = ToTensor()
        self.detector = model(**kwargs)
        self.detector.eval()

    def test(self, path, output=False, show=False):
        if show:
            cv2.namedWindow(show)
        for image, name in data.test_set(path):
            num = [0] * 49
            bls = self.detector([self.tensor(image)])[0][0]
            bls = [bls["boxes"], bls["labels"], bls["scores"]]
            boxes, labels, scores = list(map(lambda x: x.detach().numpy().tolist(), bls))
            for i in range(len(boxes)):
                box, label, score = boxes[i], labels[i], scores[i]
                if score < 0.5:
                    continue
                if show:
                    text = num2code[label]
                    score = str(int(score * 100) / 100)[:4]
                    text += "(" + score + ")"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    x1, y1, x2, y2 = list(map(int, box))
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
                    cv2.putText(image, text, (x1, y1), font, 0.4, (0, 0, 255))
                num[label] += 1
            if show:
                cv2.imshow(show, image)
                cv2.waitKey()
            if output:
                out = ["START"]
                for i in range(49):
                    if num[i]:
                        text = "Goal_ID=" + num2code[i] + ";Num=" + str(num[i])
                        out.append(text)
                out.append("END")
                with open("output/" + name + ".txt", "w") as txt:
                    txt.write("\n".join(out))
        if show:
            cv2.destroyAllWindows()


tester = Tester(
    detector.CascadeRCNN,
    number=49, iou1=0.5, iou2=0.6, iou3=0.7,
    rpn="parameters/rpn.pkl",
    net0="parameters/net.pkl",
    cascade="parameters/cascade.pkl",
    tail="parameters/tail.pkl"
)

if __name__ == "__main__":
    tester.test("data/test", output=False, show="output")
