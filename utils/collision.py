"""
Collision detection class
"""



class CollisionDetection():

    # def __init__(self):
    #     print(dir(self))

    def erHalfMtxPts(self, ellipse, rectangle):
        # er prefix stands for ellipse rectangle
        collision = False
        halfmtxpts = ellipse.getHalfMtxPts()
        for i in range(halfmtxpts.shape[1]):
            if rectangle.inPoly(halfmtxpts[:, i]):
                collision = True
                break
        return collision

    def selectCollisionChecker(self, name):
        # can use dir, or such so don't need to maintain dictionary of all methods
        if name == 'erHalfMtxPts':
            return self.erHalfMtxPts
