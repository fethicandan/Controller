import time

class PID:
    def __init__(self, P=0.2, I=0.0, D=0.0, feedback_value_last = 0.0, filtered_value_last = 0.0, ufiltered = 0.0):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time
        self.feedback_value_last = feedback_value_last
        self.filtered_value_last = filtered_value_last
        self.filtered_value = 0.0
        self.ufiltered = ufiltered
        self.clear()


    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.feedback_value_last = 0.0
        self.filtered_value_last = 0.0
        self.m1 =1
        self.m2 =1
        # Windup Guard
        self.int_error = 0.0
        self.umax=10
        self.windup_guard = 20.0

        self.output = 0.0
        self.ufiltered = 0.0
        self.filteredoutput = 0.0
        self.filtered_value = 0.0
    def update(self, feedback_value):

        filtered_value = round(feedback_value) * self.m1 + (1 - self.m1) * self.filtered_value_last
        error = (self.SetPoint - round(filtered_value))

        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_feedback= filtered_value-self.filtered_value_last

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_feedback / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error
            self.feedback_value_last = round(feedback_value, 2)
            self.filtered_value_last = round(filtered_value, 2)
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
            self.filteredoutput = self.ufiltered*(1-self.m2) + self.output * self.m2
            self.filtered_value = filtered_value
            self.ufiltered = self.filteredoutput


    def setKp(self, proportional_gain):
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        self.Kd = derivative_gain

    def setWindup(self, windup):
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        self.sample_time = sample_time