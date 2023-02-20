import json

class BombyxsimConf(object):
    def __init__(self, config_file):
        with open(config_file) as f:
            conf = json.load(f)

        self.FPS = conf['FPS']
        self.agent_type = conf['agent_type']
        self.env_type = conf['env_type']
        self.controller_type = conf['controller_type']
        self.initial_pose = conf['init_pose']
        self.random_start = conf['random_start']
        # self.hit_noise = conf['hit_noise']
        # self.hit_prob = conf['hit_probability']
        self.init_pose_epstype = conf['init_pose_eps_type']
        self.init_pose_eps = conf['init_pose_eps']
        self.wind_angle = conf['wind_angle']
        self.field_geometry = conf['field']
        self.infotaxis = conf['infotaxis']


class FieldConf(object):
    def __init__(self, conf):
        self.field_conf = conf.field_geometry
        self.source_pos = conf.field_geometry['srcpos']
        self.xlim = conf.field_geometry['xlim']
        self.ylim = conf.field_geometry['ylim']
        self.width = conf.field_geometry['width']
        self.height = conf.field_geometry['height']
        self.goalr = conf.field_geometry['goalr']


class InfotaxisConf(object):
    def __init__(self, conf):
        self.infotaxis_conf = conf.infotaxis
        self.V = conf.infotaxis['wind_speed']
        self.D = conf.infotaxis['diffusivity']
        self.E = conf.infotaxis['emission_rate']
        self.tau = conf.infotaxis['particle_lifetime']
        self.grid_shape = conf.infotaxis['grid_shape']
        self.agent_size = conf.infotaxis['agent_size']
        self.agent_speed = conf.infotaxis['agent_speed']
        self.src_radius = conf.infotaxis['src_radius']