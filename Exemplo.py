import cv2
import pytesseract

def encontrarRoiPlaca(source):
    img = cv2.imread(source)
    cv2.imshow("img", img)

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bin = cv2.threshold(cinza, 200, 255, cv2.THRESH_BINARY_INV)

    desfoque = cv2.GaussianBlur(cinza, (7, 7), 0)

    contornos, hierarquia = cv2.findContours(desfoque, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        if perimetro > 120:
            aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            if len(aprox) == 4:
                (x, y, alt, lar) = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + alt, y + lar), (0, 255, 0), 2)
                roi = img[y:y + lar, x:x + alt]
                cv2.imwrite('output/roi.png', roi)

    cv2.imshow("contornos", img)


def preProcessamentoRoiPlaca():
    img_roi = cv2.imread("output/roi.png")

    if img_roi is None:
        return

    resize_img_roi = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    img_cinza = cv2.cvtColor(resize_img_roi, cv2.COLOR_BGR2GRAY)

    _, img_binary = cv2.threshold(img_cinza, 100, 255, cv2.THRESH_BINARY)

    img_desfoque = cv2.GaussianBlur(img_binary, (5, 5), 0)

    cv2.imwrite("output/roi-ocr.png", img_desfoque)

    return img_desfoque


def ocrImageRoiPlaca():
    image = cv2.imread("output/roi-ocr.png")

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    saida = pytesseract.image_to_string(image, lang='eng', config=config)

    return saida


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))

            encontrarRoiPlaca(img_name)

            pre = preProcessamentoRoiPlaca()

            ocr = ocrImageRoiPlaca()

            print(ocr)

            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

    result, image = cam.read()

    cv2.imwrite("placa-camera.png", image)

    encontrarRoiPlaca("placa-camera.png")

    pre = preProcessamentoRoiPlaca()

    ocr = ocrImageRoiPlaca()

    print(ocr)
