import numpy as np

# let tmp be Ai-BiKi
# this is an example taken from a run
# blows up ellipse completely by 3rd timestep (E2)
tmp = np.array([[  1.00000000e+00, 0.00000000e+00, -5.28631232e-01, 2.36509661e-02, 0.00000000e+00],
                [  0.00000000e+00, 1.00000000e+00, 2.83811593e-01, 4.40526027e-02, 0.00000000e+00],
                [  0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 1.12975423e+01, -3.06328710e+04],
                [ -1.24885442e-02, -6.32984361e-03, 3.17237803e-02, 1.29081693e+00, -8.24631365e+02],
                [ -1.42749500e-01, -5.36707613e-02, 3.54734576e-01, 3.27883569e+00, -9.02123190e+03]])
E = 0.01*np.eye(5)
#print(tmp.T)
#print(E.dot(tmp.T))
E1 = tmp.dot(E).dot(tmp.T)
print(E1)
E2 = tmp.dot(E1).dot(tmp.T)
print(E2)
