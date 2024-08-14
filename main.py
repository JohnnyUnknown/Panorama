import cv2
from PIL import Image
import  math
from decimal import *
import numpy as np

def rgba_to_grayscale_with_alpha(img):
    # Проверяем, что изображение имеет 4 канала (RGBA)
    if img.shape[2] != 4:
        raise ValueError("Input image must have 4 channels (RGBA).")
    # Извлекаем альфа-канал
    alpha_channel = img[..., 3]
    # Преобразуем RGB в градации серого (используем только первые 3 канала)
    gray_image = cv2.cvtColor(img[..., :3], cv2.COLOR_RGBA2GRAY)
    # Создаем новую картинку с альфа-каналом
    gray_with_alpha = cv2.merge((gray_image, gray_image, gray_image, alpha_channel))
    return gray_with_alpha

# def transparent_edges():
#     image = Image.open(f"panorama.png").convert("RGBA")
#     # Получаем данные пикселей
#     data = image.getdata()
#     new_data = []
#     threshold = 1
#     for item in data:
#         # item - это кортеж (R, G, B, A)
#         # Если пиксель черный (R, G, B все равны 0), делаем его прозрачным (A = 0)
#         if item[0] < threshold and item[1] < threshold and item[2] < threshold:
#             new_data.append((0, 0, 0, 0))  # Прозрачный пиксель
#         else:
#             new_data.append((item[2], item[1], item[0], 255))  # Оставляем пиксель без изменений
#     # Обновляем данные изображения
#     image.putdata(new_data)
#     image = np.array(image)
#     # image = rgba_to_grayscale_with_alpha(image)
#     return image

def detect_and_match_features(img1, img2):
    # orb = cv2.ORB_create()
    # keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    # keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    #
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(descriptors1, descriptors2)
    # matches = sorted(matches, key=lambda x: x.distance)

    sift = cv2.SIFT_create()
    keypoints1, des1 = sift.detectAndCompute(img1, None)
    keypoints2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append(m)

    return keypoints1, keypoints2, good

def estimate_homography(keypoints1, keypoints2, matches, threshold=3):
    src_points = np.float32([keypoints2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    if len(src_points) < 4 and len(dst_points) < 4:
        raise Exception("err")
    print(f"src:{len(src_points)}, dst:{len(dst_points)}")

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)

    return H, mask

def warp_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # print(f"{h1=} {w1=}")
    # cv2.imshow("img_new undo", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    # Ht_temp = np.array([[1, 0, t_all[0]], [0, 1, t_all[1]], [0, 0, 1]])
    #
    # warped_img2_tmp = cv2.warpPerspective(img2, Ht_temp @ H, (xmax, ymax))
    warped_img2 = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    # cv2.imshow("img_new after", warped_img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("img_temp", warped_img2_tmp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return warped_img2, t

def image_overlay(image, warped_img, t):
    h1, w1 = image.shape[:2]

    # Назначаем альфа-канал
    alpha_channel = image[:, :, 3] / 255.0  # Преобразуем 0-255 в 0-1

    # image = transparent_edges()

    # cv2.imshow("image", warped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Наложение warped_img2 и image
    for c in range(4):  # Для каждого канала (R, G, B)
        warped_img[t[1]:h1 + t[1], t[0]:w1 + t[0], c] = (
                alpha_channel * image[:, :, c] +
                (1 - alpha_channel) * warped_img[t[1]:h1 + t[1], t[0]:w1 + t[0], c]
        )
    # print(warped_img2.shape)
    # cv2.imshow("Panorama", warped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return warped_img



path_video = 'C:\\My\\Projects\\images\\Bol2.mp4'
cap = cv2.VideoCapture(path_video)
pano = []
image_prev = []
image_new = []
frame_count = 0  # int(cap.get(cv.CAP_PROP_FPS))
# t_all = [0, 0]
stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)

while cap.isOpened():
    frame_count += 1
    ret, frame = cap.read()

    if not ret:
        print("Конец видеофайла.")
        break
    if frame_count == 120:
        frame_count = 0
        frame = cv2.resize(frame, (1024, 576))
        # frame = frame[int(480/4):int(480-480/4), int(854/6):int(854-854/6)]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        # frame = rgba_to_grayscale_with_alpha(frame)
        if len(pano) == 0 and len(image_prev) == 0:
            print("new pano")
            pano = frame.copy()
            cv2.imwrite(f"panorama.png", frame)
            image_prev = frame.copy()
            continue

        image_new = frame.copy()

        print(f"---"*10)

        keypoints1, keypoints2, matches = detect_and_match_features(image_prev, image_new)
        print(f"kp1 {len(keypoints1)}, kp2 {len(keypoints2)}")

        try:
            H, mask = estimate_homography(keypoints1, keypoints2, matches)
        except Exception:
            continue

        image_new, t_add = warp_images(image_prev, image_new, H)
        # t_all = [t_all[i] + t_add[i] for i in range(2)]

        image_prev = image_new.copy()

        pano = image_overlay(pano, image_prev, t_add)

        cv2.imwrite(f"panorama.png", pano)


cv2.imshow("Panorama", pano)
cv2.waitKey(0)
cv2.destroyAllWindows()

