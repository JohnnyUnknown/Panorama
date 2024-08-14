import cv2

path_video = 'C:\\My\\Projects\\images\\Bol2.mp4'
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

    if not ret:
        print("Конец видеофайла.")
        break
    if frame_count == 120:
        frame_count = 0
        frame = cv2.resize(frame, (1024, 576))

        # frame = frame[int(480/4):int(480-480/4), int(854/6):int(854-854/6)]
        im_list.append(frame)

        print(f"---"*10)
        if len(im_list) == 5:
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
                # break

stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
if len(im_list) > 0:
    pano_list += im_list
status, panorama = stitcher.stitch(pano_list, cv2.Stitcher_SCANS)
print(len(pano_list))
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

