import numpy as np

class Line():
    def __init__(self,n):
        self.n = n
        self.detected = False
        # x = A*y^2 + B*y + C
        # mỗi A,B,C là mỗi một list có độ dài tối đa n
        self.A = []
        self.B = []
        self.C = []
        # tính trung bình mỗi giá trị
        self.A_avg = 0.
        self.B_avg = 0.
        self.C_avg = 0.

    def get_fit(self):
        return (self.A_avg,self.B_avg,self.C_avg)

    def add_fit(self,fit_coeffs):
        # fit_coeffs là danh sách 3 phần tử của hệ số của đa thức bậc 2
        # hệ số xếp hàng đầy
        q_full = len(self.A) >= self.n
        self.A.append(fit_coeffs[0])
        self.B.append(fit_coeffs[1])
        self.C.append(fit_coeffs[2])

        if q_full:
            _ = self.A.pop(0)
            _ = self.B.pop(0)
            _ = self.C.pop(0)

        # trung bình của hệ só dòng
        self.A_avg = np.mean(self.A)
        self.B_avg = np.mean(self.B)
        self.C_avg = np.mean(self.C)

        return (self.A_avg, self.B_avg, self.C_avg)