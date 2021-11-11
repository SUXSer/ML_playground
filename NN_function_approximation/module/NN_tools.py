from numpy import asarray
from sklearn.preprocessing import MinMaxScaler

def data_setup(func):
    x = asarray([i for i in range(-50,51)])
    y = asarray([func(i) for i in x])
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    scale_x = MinMaxScaler()
    x = scale_x.fit_transform(x)
    scale_y = MinMaxScaler()
    y = scale_y.fit_transform(y)

    return x, y, scale_x, scale_y
