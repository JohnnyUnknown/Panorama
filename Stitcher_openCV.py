import cv2

path_video = 'C:\\My\\Projects\\images\\move3.mp4'
cap = cv2.VideoCapture(path_video)
pano = []
panorama = None
frame_count = 0  # int(cap.get(cv.CAP_PROP_FPS))
im_list = []
pano_list = []
stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)

while cap.isOpened():
    frame_count += 1
    ret, frame = cap.read()
    if len(pano_list) == 3:
        break

    if not ret:
        print("Конец видеофайла.")
        break
    if frame_count == 25:
        frame_count = 0
        frame = cv2.resize(frame, (1920, 1080))

        im_list.append(frame)

        print(f"---"*10)
        if len(im_list) == 3:
            status, pano = stitcher.stitch(im_list, cv2.Stitcher_SCANS)
            if status == cv2.Stitcher_OK:
                print("OK")
                im_list.clear()
                pano_list.append(pano)
            else:
                if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
                    print("Недостаточно изображений для сшивания.")
                elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
                    print("Ошибка при оценке гомографии.")
                else:
                    print("Ошибка во время сшивания:", status)
                break

stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
# if len(im_list) > 0:
#     print("undo", len(pano_list))
#     pano_list += im_list
#     print("after", len(pano_list))
status, panorama = stitcher.stitch(pano_list, cv2.Stitcher_SCANS)
if status == cv2.Stitcher_OK:
    print("OK Panorama")
    cv2.imwrite(f"pano_stitch.png", panorama)
else:
    if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("Недостаточно изображений для сшивания.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("Ошибка при оценке гомографии.")
    else:
        print("Ошибка во время сшивания:", status)


cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()

