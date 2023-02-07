import gymnasium as gym
import os
from evdev import ecodes, InputDevice
import select
import pickle


class TR300Control:
    def __init__(self, name):
        self.dev_name = name
        self.input_device = None

    def get_input_device(self):
        if self.input_device is None or self.input_device.fd == -1:
            if os.access(self.dev_name, os.R_OK):
                self.input_device = InputDevice(self.dev_name)
        return self.input_device

    def read_events(self, timeout):
        input_device = self.get_input_device()
        if input_device is not None and input_device.fd != -1:
            r, _, _ = select.select({input_device.fd: input_device}, [], [], timeout)
            if input_device.fd in r:
                for event in input_device.read():
                    yield event
    def set_autocenter(self, autocenter):
        if autocenter > 100:
            autocenter = 100
        autocenter = str(int(autocenter / 100.0 * 65535))
        print(f"Setting autocenter strength: {autocenter}")
        input_device = self.get_input_device()
        ret = input_device.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, int(autocenter))
        #ret = input_device.write(ecodes.EV_FF, ecodes.FF_CONSTANT, 10000)
        print(ret)
tr300_control = TR300Control("/dev/input/by-id/usb-Thrustmaster_Thrustmaster_T300RS_Racing_wheel-event-joystick")
tr300_control.set_autocenter(20)

ENV_NAME = "CarRacing-v2"
env = gym.make(ENV_NAME, render_mode='human')

runs = 280
while True:
    s, _ = env.reset()
    env.render()
    throttle = 0
    steer = 0
    brake = 0
    total_reward = 0
    steps = 0

    #os.makedirs(f"tracks/{runs:04d}/", exist_ok=False)

    traj_data = []
    while True:
        g = tr300_control.read_events(0.01)
        h = 65536/2
        for data in g:
            #print(data, data.code, data.type, data.value)
            if data.type == 3: #axises
                if data.code == 0:
                    steer = (data.value - h) / h
                elif data.code == 2:
                    throttle = (1024 - data.value ) / 1024
                    if throttle < 0.01:
                        throttle = 0
                elif data.code == 5:
                    brake = (1024 - data.value ) / 1024
                    if brake < 0.01:
                        brake = 0
        action = [steer, throttle, brake]
        sp, r, done, truncated, info = env.step(action)
        traj_data.append((s, action, r, done, truncated))

        done = done or truncated
        env.render()

        s = sp

        steps += 1
        if done:
            with open(f'tracks/{runs:04d}.pickle', 'wb') as handle:
                pickle.dump(traj_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            break
        total_reward += r
    print(f"#{runs} done, steps:{steps}, total_reward:{total_reward}")
    runs += 1


