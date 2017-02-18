import numpy as np


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # how many times was line not detected in a row?
        self.grace_amount = 0
        # coefficient values of the last n fits of the line
        self.recent_fitted = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = 0
        # distance in meters of vehicle center from the line
        self.line_base_pos = 0
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # image indices for detected line pixels
        self.indx = None

    def should_update(self, curvature, fit):
        # use differences in curvature between previous and new to determine update
        curve_compare = abs(self.radius_of_curvature - curvature)
        intersect_compare = np.sqrt((self.best_fit[2]-fit[2])**2)
        slope_compare = np.sqrt((self.best_fit[0]-fit[0])**2) + np.sqrt((self.best_fit[1]-fit[1])**2)
        # print(curve_compare, intersect_compare, slope_compare)
        if curvature < 1500 or slope_compare < .35:
            self.radius_of_curvature = curvature
            return True
        else:
            print(curvature, slope_compare)
            # self.grace_amount += 1
            return False

    def should_update_blind(self, curvature, fit):
        if len(self.best_fit) > 0:
            if curvature < 1500:
                self.radius_of_curvature = curvature
                return True
            else:
                # self.grace_amount += 1
                return False
        else:
            return True

    def update(self, fit, indx, x, y):
        self.current_fit = fit
        self.indx = indx
        self.allx = x
        self.ally = y
        self.track_fits()

    def track_fits(self):
        if len(self.recent_fitted) < 6:
            self.recent_fitted.append(self.current_fit)
        else:
            self.recent_fitted.pop()
            self.recent_fitted.append(self.current_fit)
        self.avg_fits()

    def avg_fits(self):
        best = [0, 0, 0]
        for fit in self.recent_fitted:
            best += fit
        self.best_fit = best/float(len(self.recent_fitted))
