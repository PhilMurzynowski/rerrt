"""
Collision detection
"""


class CollisionDetection():
    """
    Class designed to contain selection of collision detection methods.
    Currently supports:
        ellipse-rectangle collision halfmtx points heuristic
    Note: GJK Algorithm for general convex polygons coming shortly..
    """


    def erHalfMtxPts(self, ellipse, rectangle):
        """er prefix stands for ellipse-rectangle
        Given Ellipse and Rectangle objects, checks if the columns
        of the square root of the matrix describing the ellipse
        are within the rectangle.
        """
        collision = False
        halfmtxpts = ellipse.getHalfMtxPts()
        for i in range(halfmtxpts.shape[1]):
            if rectangle.inPoly(halfmtxpts[:, i]):
                collision = True
                break
        return collision

    def selectCollisionChecker(self, name):
        """Selection tool to return desired collision detection method
        Note: Can update to use dir, or such so don't need to maintain dictionary of all methods
        """
        if name == 'erHalfMtxPts':
            return self.erHalfMtxPts
