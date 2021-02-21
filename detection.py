import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single
from functools import cmp_to_key

def contour_sort(a, b):
    br_a = cv2.boundingRect(a)
    br_b = cv2.boundingRect(b)

    if abs(br_a[1] - br_b[1]) <= 15:
        return br_a[0] - br_b[0]

    return br_a[1] - br_b[1]

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
img_path = "test/test8.jpg"

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Đọc file ảnh đầu vào
Ivehicle = cv2.imread(img_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

_ , LpImg, lp_type, Coodinates = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

model_svm = cv2.ml.SVM_load('svm.xml')

if (len(LpImg)):

    # Chuyen doi anh bien so
    LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    roi = LpImg[0]

    # Chuyen anh bien so ve gray
    gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)

    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(gray, 100, 255,
                         cv2.THRESH_BINARY_INV)[1]

    #cv2.imshow("Anh bien so sau threshold", binary)
    #cv2.waitKey()

    # Segment kí tự
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key=cmp_to_key(contour_sort))

    if float(binary.shape[1] / binary.shape[0]) >= 3:
        flag = 1
    else:
        flag = 2
    plate_info = ""
    for c in cont:
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if ((1.5<=ratio<=5) and (flag == 1) and h/roi.shape[0]>=0.6) \
                or ((1.5<=ratio<=5) and (flag == 2) and h/roi.shape[0]>=0.3):
            print(x, y, w, h)
            # Ve khung chu nhat quanh so
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Tach so va predict
            curr_num = thre_mor[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
            curr_num = np.array(curr_num,dtype=np.float32)
            curr_num = curr_num.reshape(-1, digit_w * digit_h)

            # Dua vao model SVM
            result = model_svm.predict(curr_num)[1]
            result = int(result[0, 0])

            if result<=9: # Neu la so thi hien thi luon
                result = str(result)
            else: #Neu la chu thi chuyen bang ASCII
                result = chr(result)

            plate_info +=result

    #cv2.imshow("Cac contour tim duoc", roi)
    #cv2.waitKey()

    # Viet bien so len anh
    cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

    # Hien thi anh
    print("Bien so=", plate_info)
    color = (255,0,0)
    thickness = 2
    cv2.rectangle(Ivehicle, (int(Coodinates[0][0][0]),int(Coodinates[0][1][0])), (int(Coodinates[0][0][2]), int(Coodinates[0][1][2])), color, thickness)
    cv2.imshow("Detected",Ivehicle)
    cv2.waitKey()

cv2.destroyAllWindows()
