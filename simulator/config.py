tlim = 250              # Simulation time limit in seconds
env = "smokevid"        #Type of environment
agt = "silkmoth"        #Type of agent
controller = ['KPB']    #Type of controller: [KPB, IRL (specify policy file)]
Nruns = 1              #Number of simulation runs
animation = False       #Draw animation
input_dir = "bombyxsim-template" # Path of the directory with odor plume data
save_log = True
plt_traj = True

smoke_environment = {
    "FPS": 30,
    "env": {
        "srcpos": [0, 0],
        "xspace": [0, 600],
        "yspace": [-360, 360],
        "width": 512,
        "height": 640,
        "goalr": 50
    },
    "irl":{
        "num_states": ["tblank", "hits"]
    },
    "init_pose": [300.0, 0.0, 180.0],
    "random_start": True,
    "init_pose_eps": [
        10.0,
        10.0,
        30.0
    ],
    "hit_noise": True,
    "hit_probability": 1.0,
    "wind_angle": 0.0
}