import RPi.GPIO as GPIO
import time



class Control():
    def __init__(self):
        # Khai báo ngõ GPIO
        self.IN1 = 16
        self.IN2 = 20
        self.ENA = 12
        self.IN3 = 6
        self.IN4 = 5
        self.ENB = 13
        GPIO.setmode(GPIO.BCM)  # khai báo chân kiểu BCM
        GPIO.setwarnings(False)  # tắt thông báo GPIO

        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)
        GPIO.setup(self.IN3, GPIO.OUT)
        GPIO.setup(self.IN4, GPIO.OUT)
        GPIO.setup(self.ENA, GPIO.OUT)
        GPIO.setup(self.ENB, GPIO.OUT)

        self.PWMA = GPIO.PWM(self.ENA, 500)
        self.PWMB = GPIO.PWM(self.ENB, 500)

        self.PWMA.start(0)
        self.PWMB.start(0)

    def STOP(self):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
        self.PWMA.ChangeDutyCycle(0)
        self.PWMB.ChangeDutyCycle(0)

    def RUN(self):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.PWMA.ChangeDutyCycle(50)
        self.PWMB.ChangeDutyCycle(50)

    def LEFT(self):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.PWMA.ChangeDutyCycle(0)
        self.PWMB.ChangeDutyCycle(68)
        time.sleep(2.5)

    def RIGHT(self):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
        self.PWMA.ChangeDutyCycle(68)
        self.PWMB.ChangeDutyCycle(0)
        time.sleep(2.5)


if __name__ == '__main__':
    Control = Control()
    Control.RUN()
    time.sleep(2.5)
    Control.STOP()