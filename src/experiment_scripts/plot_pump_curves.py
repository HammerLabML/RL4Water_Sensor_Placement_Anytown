import numpy as np
import matplotlib.pyplot as plt

def pump_head_gain(flow, shutoff_head, pump_curve_coeffs, speed_setting):
    r, n = pump_curve_coeffs
    head_gain = speed_setting**2 * (shutoff_head - r * (flow / speed_setting)**n)
    return head_gain

r = 0.5
n = 2
shutoff_head = 1
flows = np.linspace(0, 2, 200)
for speed_setting in [.5, 1, 1.5, 2]:
    plt.plot(
        flows,
        pump_head_gain(flows, shutoff_head, [r, n], speed_setting),
        label=f'Speed: {speed_setting}'
    )
plt.grid(True)
plt.legend()
plt.show()

