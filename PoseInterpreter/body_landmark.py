import json


class PoseInterpreterBodyLandmark:

    def __init__(self, identifier, name, x=None, y=None, z=None, visibility=0, angle=None, z_angle=None,
                 math_angle=None):
        self.id = identifier
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.angle = angle
        self.z_angle = z_angle
        self.math_angle = math_angle

    def __str__(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return json.dumps(self.__dict__)
