import cv2
import numpy as np
import pickle
from Line import Line
#import matplotlib.pyplot as plt

# canh chỉnh lại ảnh đầu vào của camera
def undistort(img):
    with open('calibrate_camera.p', 'rb') as f:
        save_dist = pickle.load(f)
        mtx = save_dist['mtx']
        dist = save_dist['dist']
        frame = cv2.undistort(img, mtx, dist, None, mtx)
    return frame

# Bước Perspective Transformation chuyển từ góc nhìn thứ nhất sang góc nhìn chim bay
def warp_img(img):
     # tạo 2 thông số cần thiết
    src = np.float32([[ 696 , 455 ],[ 587 , 455 ],[ 235 , 700 ],[ 1075 , 700 ]])
    dst = np.float32([[ 930 , 0 ],[350 , 0 ],[ 350 , 720 ],[930 , 720 ]])
    # tính ma trận biến đổi m và nghịch đảo của m
    m = cv2.getPerspectiveTransform(src,dst)
    m_inv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img,m,(img.shape[1],img.shape[0]),flags= cv2.INTER_LINEAR)
    un_warped = cv2.warpPerspective(warped,m_inv,(warped.shape[1],warped.shape[0]),flags=cv2.INTER_LINEAR)
    return warped, un_warped, m, m_inv

# Bước tạo ngưỡng để lấy cạnh đường
def abs_sobel_thresh(img, orient = 'x',thresh_min = 20, thresh_max = 100):
    # chuyển ảnh đầu vào thành ảnh xám
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Áp dụng gradient x hoặc y với hàm OpenCV Sobel () và lấy giá trị tuyệt đối
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0)) # lọc sobel tìm cạnh theo hướng gradient x
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F,0,1)) # lọc sobel tìm cạnh theo hướng gradient y
    # chuyển lại thành định dạng số nguyên 8bit
    scale_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # tạo ảnh nhị phân dể áp vào ảnh sobel
    binary_output = np.zeros_like(scale_sobel)
    # cho những điểm ảnh thỏa đk bằng giá trị 255
    binary_output[(scale_sobel>= thresh_min) & (scale_sobel <= thresh_max)] = 1

    return binary_output

def mag_thresh(img, sobel_kernel = 3,mag_thresh = (30,100)):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize= sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize= sobel_kernel)
    gradmag  = np.sqrt(sobelx**2 + sobely**2)
    # chuyển về type 8bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output

def dir_thresh(img, sobel_kernel = 3, thresh = (0, np.pi/2)):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize= sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize= sobel_kernel)
    absgraddir = np.arctan2(np.abs(sobely), np.abs(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output

def hls_thresh(img,thresh = (100,255)):
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    channel = hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

    return binary_output

def combined_thresh(img):
    abs_bin = abs_sobel_thresh(img, orient= 'x', thresh_min= 50, thresh_max= 255)
    mag_bin = mag_thresh(img,sobel_kernel= 3, mag_thresh=(50,255))
    dir_bin = dir_thresh(img,sobel_kernel= 15, thresh= (0.7,1.3))
    hls_bin = hls_thresh(img,thresh=(170,255))

    combined = np.zeros_like(dir_bin)
    combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1] = 1

    return combined

# dùng kỹ thuật sliding window để tìm kiếm tọa độ của làn đường sau đó chuyển chúng thành một hàm bậc 2 với hàm polyfit
def line_fit(binary_img,rgb_img):
    '''
    tìm kiếm và điều chỉnh các làn đường
    '''
    # khi đã có ảnh đầu vào xác định rõ được làn đường
    # vẽ histogram cho nửa dưới của ảnh đầu vào
    histogram = np.sum(binary_img[binary_img.shape[0]//2:,:], axis = 0) # tính tổng giá trị của từng cột
    # tạo hình ảnh đầu ra để vẽ và hình dung kết quả
    out_img = (np.dstack((binary_img,binary_img,binary_img))*255).astype('uint8') # tạo ảnh có 3 kênh màu từ ảnh binary_img chỉ có 1 kênh màu
    # tìm đỉnh của nửa bên trái vầ bên phải biểu đồ
    # đây sẽ là điểm bắt đầu cho các dòng bên trái và bên phải
    mid_point = histogram.shape[0]//2
    leftx_base = np.argmax(histogram[:mid_point]) # lấy giá trị lớn nhất của phần ảnh bên trái
    rightx_base = np.argmax(histogram[mid_point:]) + mid_point
    # chọn số lượng cửa sổ trượt
    nwindow = 9
    # tính chiều cao cho mỗi cửa sổ
    window_height = np.int(binary_img.shape[0]/nwindow)
    # tìm tất cả các điểm x,y có giá trị khác 0
    nonzero = binary_img.nonzero()
    nonzero_y = np.array(nonzero[0]) # hàm trả về một mảng các tọa dộ y có giá trị khác 0
    nonzero_x = np.array(nonzero[1]) # hàm trả về một mảng các tọa độ x có giá trị khác 0
    # vị trí hiện tại được cập nhật sau mỗi vòng lặp frame
    leftx_curent = leftx_base
    rightx_curent = rightx_base
    # đặt chiều rộng của mỗi cửa sổ +/- margin
    margin = 100
    # đặt số lượng điểm ảnh tối thiểu tìm được cho cửa sổ tiếp theo
    minpix = 50
    # tạo một list để lưu trữ các điểm ảnh thuộc làn đường bên trái và bên phải
    left_lane_ind = []
    right_lane_ind = []

    # lặp tất cả các cửa sổ
    for window in range(nwindow):
        # vẽ ranh giới của cửa sổ
        win_y_low = binary_img.shape[0] - (window + 1)*window_height
        win_y_high = binary_img.shape[0] - window*window_height
        win_xleft_low = leftx_curent - margin
        win_xleft_high = leftx_curent + margin
        win_xright_low = rightx_curent - margin
        win_xright_high = rightx_curent + margin
        # vẽ cửa sổ
        cv2.rectangle(rgb_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 0), 2)
        cv2.rectangle(rgb_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 0), 2)
        # tìm các điểm ảnh có giá trị khác 0 khác trong cửa sổ
        good_left_ind = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & \
                        (nonzero_x >= win_xleft_low) & (nonzero_x <= win_xleft_high)).nonzero()[0]
        good_right_ind = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & \
                        (nonzero_x >= win_xright_low) & (nonzero_x <= win_xright_high)).nonzero()[0]
        # thêm các chỉ số về làn bên phải và bên trái vào list trước đó
        left_lane_ind.append(good_left_ind)
        right_lane_ind.append(good_right_ind)
        # khi đã tìm đủ các giá trị lớn hơn số lượng minpix, tạo độ x của cửa sổ tiếp theo sẽ là giá trị trung bình của các điểm đó
        if len(good_left_ind) > minpix:
            leftx_curent = np.int(np.mean(nonzero_x[good_left_ind]))
        if len(good_right_ind) > minpix:
            rightx_curent = np.int(np.mean(nonzero_x[good_right_ind]))

    # nối các điểm đã tìm được
    left_lane_ind = np.concatenate(left_lane_ind)
    right_lane_ind = np.concatenate(right_lane_ind)
    # trích xuất tạo độ các điểm ảnh thuộc làn bên phải và bên trái
    leftx = nonzero_x[left_lane_ind]
    lefty = nonzero_y[left_lane_ind]
    rightx = nonzero_x[right_lane_ind]
    righty = nonzero_y[right_lane_ind]
    # tìm một đa thức bật 2 cho mỗi làn đường
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    # trả về một dict các biến có liên quan
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzero_x'] = nonzero_x
    ret['nonzero_y'] = nonzero_y
    ret['out_img'] = out_img
    ret['left_lane_ind'] = left_lane_ind
    ret['right_lane_ind'] = right_lane_ind

    return ret

# tối ưu làn đường tìm được
def tune_fit(binary_img,left_fit,right_fit):
    nonzero = binary_img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    left_lane_ind = ((nonzero_x > (left_fit[0]*(nonzero_y**2) + left_fit[1]*nonzero_y + left_fit[2] - margin)) & \
                         (nonzero_x < (left_fit[0]*(nonzero_y**2) + left_fit[1]*nonzero_y + left_fit[2] + margin)))
    right_lane_ind = ((nonzero_x > (right_fit[0]*(nonzero_y**2) + right_fit[1]*nonzero_y + right_fit[2] - margin)) & \
                         (nonzero_x < (right_fit[0]*(nonzero_y**2) + right_fit[1]*nonzero_y + right_fit[2] + margin)))
    # trích xuất các pixel bên trái và bên phải
    leftx = nonzero_x[left_lane_ind]
    lefty = nonzero_y[left_lane_ind]
    rightx = nonzero_x[right_lane_ind]
    righty = nonzero_y[right_lane_ind]
    # nếu không tìm đủ số lượng pixel thì trả về none
    min_ind = 10
    if lefty.shape[0] < min_ind or righty.shape[0] < min_ind:
        return None
    # tạo một hàm bậc 2 với các tọa độ vừa tìm được
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    # tạo các giá trị x,y để vẽ làn đường
    ploty = np.linspace(0,binary_img.shape[0]-1,binary_img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fity = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # lưu các thông số cần thiết vào một dict dữ liệu
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzero_x'] = nonzero_x
    ret['nonzero_y'] = nonzero_y
    ret['left_lane_ind'] = left_lane_ind
    ret['right_lane_ind'] = right_lane_ind

    return ret

# tính độ lệch từ của làn đường so với tâm xe theo hệ met
def calc_center_offset(undist,left_fit,right_fit):
    # tính độ lệch theo pixel
    bottom_y = undist.shape[0] - 1
    bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
    bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_x_left + right_fit[2]
    center_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2
    # chuyển từ pixel sang met
    xm_per_pix = 3.7/700
    center_offset = xm_per_pix*center_offset
    center_offset = round(center_offset,1)

    return center_offset

# vẽ làn đường lên ảnh gốc
def result_viz(undist,left_fit,right_fit,m_inv, center_offset):
    # tạo các giá trị x,y để vẽ
    ploty = np.linspace(0,undist.shape[0]-1,undist.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1] * ploty + right_fit[2]

    # tạo hình ảnh để vẽ lên
    color_warp = np.zeros((720,1280,3), dtype = 'uint8')

    # viết lại định dạng x,y để có thể phù hợp với hàm cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx , ploty])))])
    pts = np.hstack((pts_left,pts_right))

    # vẽ làn đường vào hình ảnh trống đã tạo
    cv2.fillPoly(color_warp,np.int_([pts]),(255,255,0))

    # chuyển phần làn đường đã vẽ từ góc nhìn chim bay sang góc nhìn thứ nhất
    new_warp = cv2.warpPerspective(color_warp,m_inv,(undist.shape[1],undist.shape[0]))
    # thêm kết quả vào ảnh gốc
    result =cv2.addWeighted(undist,1,new_warp,0.3,0)
    # ghi độ lệch và góc lệch lên ảnh
    #avg_curverad = (left_curverad + right_curverad)/2
    #cv2.putText(result,str('Radius of curvature: ') + str(avg_curverad) + str('m'),(30,40),0,1,(0,0,0),2,cv2.LINE_AA)
    if (left_fit[1] + right_fit[1])/2 > 0.05:
        cv2.putText(result, 'Left turn', (30, 40), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
    if (left_fit[1] + right_fit[1])/2 < -0.05:
        cv2.putText(result,'Right turn', (30, 40), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
    if -0.05 < (left_fit[1] + right_fit[1])/2 < 0.05:
        cv2.putText(result, 'Straight', (30, 40), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(result,str('Offset from lane center: ') + str(center_offset) + str('m'),(30,70),0,1,(0,0,0),2,cv2.LINE_AA)

    return result

if __name__ == '__main__':
    cap = cv2.VideoCapture('D:\PYTHON\XE TU HANH PROJECT\project_video.mp4')

    window_size =5
    left_line = Line(n = window_size)
    right_line = Line(n = window_size)
    detected = False
    left_curverad, right_curverad = 0., 0.
    left_lane_ind, right_lane_ind = None, None
    while True :
        success, frame = cap.read()
        # # bước chỉnh sửa ảnh
        undist = undistort(frame)
        # # bước chuyển từ góc nhìn thứ nhất sang góc nhìn chim bay
        combined = combined_thresh(undist)
        binary_warped, binary_unwarped, m, m_inv = warp_img(combined)

        if not detected:
            ret = line_fit(binary_warped, undist)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzero_x = ret['nonzero_x']
            nonzero_y = ret['nonzero_y']
            left_line_ind = ret['left_lane_ind']
            right_line_ind = ret['right_lane_ind']

            left_fit = left_line.add_fit(left_fit)
            right_fit = right_line.add_fit(right_fit)

            #left_curverad, right_curverad = calc_curva(left_lane_ind,right_lane_ind,nonzero_x,nonzero_y)
            detected = True

        else:
            left_fit = left_line.get_fit()
            right_fit = right_line.get_fit()
            ret = tune_fit(binary_warped,left_fit,right_fit)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzero_x = ret['nonzero_x']
            nonzero_y = ret['nonzero_y']
            left_line_ind = ret['left_lane_ind']
            right_line_ind = ret['right_lane_ind']

            if ret is not None:
                left_fit = ret['left_fit']
                right_fit = ret['right_fit']
                nonzero_x = ret['nonzero_x']
                nonzero_y = ret['nonzero_y']
                left_line_ind = ret['left_lane_ind']
                right_line_ind = ret['right_lane_ind']

                left_fit = left_line.add_fit(left_fit)
                right_fit = right_line.add_fit(right_fit)

                #left_curverad, right_curverad = calc_curva(left_lane_ind, right_lane_ind, nonzero_x, nonzero_y)

            else:
                detected = False
        center_offset = calc_center_offset(undist, left_fit, right_fit)
        result = undist
        result = result_viz(undist, left_fit, right_fit, m_inv,center_offset)
        cv2.imshow('result', result)
        #cv2.imshow('frame', combined)

        cv2.waitKey(1)
