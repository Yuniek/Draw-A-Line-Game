import pymunk

class Ball:
    def __init__(self, space, x, y):
        self.mass = 1
        self.radius = 10
        self.inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        
        self.body = pymunk.Body(self.mass, self.inertia)
        self.body.position = x, y
        
        self.shape = pymunk.Circle(self.body, self.radius, (0, 0))
        self.shape.elasticity = 0.8
        self.shape.friction = 0.5
        
        # Add to physics space
        space.add(self.body, self.shape)

        self.active = True

    @property
    def x(self):
        return self.body.position.x

    @property
    def y(self):
        return self.body.position.y

    def deactivate(self, space):
        if self.active:
            space.remove(self.body, self.shape)
            self.active = False