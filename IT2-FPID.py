import numpy as np
import skfuzzy as fuzz
import time


class F2PID:
    def __init__(self, alfa=0.0, beta=0.0, Ke = 0.0, Kd = 0.0, Feedback_Val = 0.0):

        self.alfa = alfa
        self.beta = beta
        self.Kd = Kd
        self.Ke = Ke
        self.delta_y = 0.0
        self.delta_error = 0.0
        self.last_feedback = Feedback_Val
        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time
        self.clear()

        self.c1 = -1
        self.c2 = self.c4 = -0.7
        self.c3 = self.c5 = self.c7 = 0
        self.c6 = self.c8 = 0.7
        self.c9 = 1
        self.output_table = np.matrix([[self.c1, self.c2, self.c3], [self.c4, self.c5, self.c6], [self.c7, self.c8, self.c9]])
        self.outputlist = np.transpose(np.asarray(self.output_table).reshape(-1))

    def clear(self):
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.windup_guard = 20.0
        self.output = 0.0
        self.last_feedback = 0.0
        self.delta_y = 0.0
        self.delta_error = 0.0

    def fuzzycode(self, E, dE):

        error = np.array([E])
        dError = np.array([dE])

        if error < np.array([-1]):
            error = np.array([-1])
        elif error > np.array([1]):
            error = np.array([1])
        else:
            error = error

        if dError < np.array([-1]):
            dError = np.array([-1])
        elif dError > np.array([1]):
            dError = np.array([1])
        else:
            dError = dError
        
        
        scale = 0.25
        theta = 0.5
        
        higher_error_N = fuzz.trimf(error, [-1.8, -1, 0])
        higher_error_Z = fuzz.trimf(error, [-1, 0, 1])
        higher_error_P = fuzz.trimf(error, [0, 1, 1.8])
        
        lower_error_N = (1 - scale) * (fuzz.trimf(error, [-1.8, -1, 0]))
        lower_error_Z = scale * (fuzz.trimf(error, [-1, 0, 1]))
        lower_error_P = (1 - scale) * (fuzz.trimf(error, [0, 1, 1.8]))
        
        higher_derror_N = fuzz.trimf(dError, [-1.8, -1, 0])
        higher_derror_Z = fuzz.trimf(dError, [-1, 0, 1])
        higher_derror_P = fuzz.trimf(dError, [0, 1, 1.8])
        
        lower_derror_N = (1 - scale) * (fuzz.trimf(dError, [-1.8, -1, 0]))
        lower_derror_Z = scale * (fuzz.trimf(dError, [-1, 0, 1]))
        lower_derror_P = (1 - scale) * (fuzz.trimf(dError, [0, 1, 1.8]))

        upper_error_matrix = np.transpose([higher_error_N, higher_error_Z, higher_error_P])
        upper_dError_matrix = np.array([higher_derror_N, higher_derror_Z, higher_derror_P])

        lower_error_matrix = np.transpose([lower_error_N, lower_error_Z, lower_error_P])
        lower_dError_matrix = np.array([lower_derror_N, lower_derror_Z, lower_derror_P])
        
        
        f_higher_Table = upper_error_matrix * upper_dError_matrix
        f_lower_Table = lower_dError_matrix * lower_error_matrix
        
        
        flist_lower = np.asarray(f_lower_Table).reshape(-1)
        flist_higher = np.asarray(f_higher_Table).reshape(-1)
        
        
        foutput_lower = np.sum(flist_lower * self.outputlist) / np.sum(flist_higher + flist_lower)
        foutput_higher = np.sum(flist_higher * self.outputlist) / np.sum(flist_higher + flist_lower)
        
        foutput = ((theta * foutput_lower)  + ((1- theta) * foutput_higher))
        # foutput = (foutput_lower + foutput_higher) / 2 # alpha - (1- alpha) şeklinde değiştir.
        return foutput

    def update(self, feedback_value):

        error = self.SetPoint - feedback_value
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_y = feedback_value - self.last_feedback
        self.delta_y = delta_y
        
        if (delta_time >= self.sample_time):
            E = error * self.Ke
            self.edot = 0.0
            if delta_time > 0:
                self.edot = delta_y / delta_time # d(r-y)/dt
                dE = self.edot * self.Kd              
                U = F2PID.fuzzycode(self, E, dE)
                self.PTerm = self.alfa * U
                self.ITerm += self.beta * (U * delta_time)
                if (self.ITerm < -self.windup_guard):
                    self.ITerm = -self.windup_guard
                elif (self.ITerm > self.windup_guard):
                    self.ITerm = self.windup_guard

                self.last_feedback = feedback_value
                self.last_time = self.current_time
                self.last_error = error
                self.output = self.PTerm + self.ITerm


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

