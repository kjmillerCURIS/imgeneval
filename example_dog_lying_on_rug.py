from yolov5 import YOLOv5
import mediapipe as mp
from PIL import Image
import numpy as np


def do_pose(imname):
    pose = mp.solutions.pose.Pose()
    image = Image.open(imname)
    image = np.array(image)
    results = pose.process(image)
    return results


def do_detection(imname):
    model = YOLOv5('yolov5s.pt')
    results = model.predict(imname)
    return results


if __name__ == '__main__':
#    results = do_detection('dog_lying_on_rug.jpg')
#    print(results)
#    import pdb
#    pdb.set_trace()

#    results = do_pose('dog_lying_on_rug.jpg')
#    print(results)
#    import pdb
#    pdb.set_trace()

    results = do_pose('dog_standing.jpg')
    print(results)
    import pdb
    pdb.set_trace()

#    results = do_pose('person_lying_on_rug.jpg')
#    print(results)
#    import pdb
#    pdb.set_trace()
