import cv2
import pathlib

# from image_processing import FacialImageProcessing
from facial_analysis import FacialImageProcessing


def rect_intersection_square(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y
    if w < 0 or h < 0:
        w = h = 0
    return w * h


def crop_faces():
    target_dir = pathlib.Path('C:\\akharche\\MasterThesis\\dataset\\ytf_faces\\')
    dataset_path = pathlib.Path('C:\\akharche\\MasterThesis\\dataset\\aligned_images_DB')
    target_classes_path = 'C:\\akharche\\MasterThesis\\lfw_ytf_classes.txt'

    metadata = 'C:\\akharche\\MasterThesis\\ytf_frames_meta_tmp.csv'

    imgProcessing = FacialImageProcessing(print_stat=False, mtcnn_detector=True, minsize=36)

    target_classes = set()
    with open(target_classes_path, mode='r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            target_classes.add(line.strip())

    print(len(target_classes))

    data_imgs = []

    with open(metadata, mode='r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split(',')
            data_imgs.append(line[2])

    for img_path in data_imgs:
        img_path = pathlib.Path(img_path)
        pts = img_path.parts
        img_class = pts[5]
        video_id = pts[6]
        img_name = img_path.name

        if str(img_class) in target_classes and str(img_class) not in 'Aaron_Sorkin':
            draw = cv2.imread(str(img_path))
            img = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = img.shape

            bounding_boxes, points = imgProcessing.detect_faces(img)

            best_bb = []
            best_square = 0
            try:
                box = bounding_boxes[0]
            except Exception:
                print("BAD: ", img_path)
                continue

            for i, bb in enumerate(bounding_boxes):

                b = [int(bi) for bi in bb]
                # y1,x1,y2,x2=b[0:4]
                if b[0] < 0: b[0] = 0
                if b[1] < 0: b[1] = 0
                if b[2] > img_w: b[2] = img_w
                if b[3] > img_h: b[3] = img_h
                if b[2] > b[0] and b[3] > b[1]:
                    sq = rect_intersection_square(b, box)
                    if sq > best_square:
                        best_square = sq
                        best_bb = b
            if len(best_bb) != 0:
                face_x, face_y = best_bb[0], best_bb[1]
                face_w, face_h = (best_bb[2] - best_bb[0]), (best_bb[3] - best_bb[1])
                dw, dh = 10, 10  # max(int(face_w*0.05),10),max(int(face_h*0.05),10)
                sz = max(face_w + 2 * dw, face_h + 2 * dh)
                dw, dh = (sz - face_w) // 2, (sz - face_h) // 2

                box = (max(0, face_x - dw), max(0, face_y - dh), min(img_w, face_x + face_w + dw),
                       min(img_h, face_y + face_h + dh))
                # box = (face_x, face_y, face_x+face_w, face_y+face_h)
                # box=best_bb

            print(str(img_path))

            area = draw[box[1]:box[3], box[0]:box[2], :]

            folder_1 = pathlib.Path(str(target_dir) + f'\\{img_class}')

            if not folder_1.exists():
                folder_1.mkdir(parents=True, exist_ok=True)

            folder_2 = pathlib.Path(str(folder_1) + f'\\{video_id}')
            if not folder_2.exists():
                folder_2.mkdir(parents=True, exist_ok=True)

            # new_dir = target_dir + f'{img_class}\\{video_id}\\{img_name}'
            new_dir = str(folder_2) + f'\\{img_name}'

            cv2.imwrite(new_dir, area)

if __name__ == '__main__':
    crop_faces()