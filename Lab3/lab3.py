import pods
import pylab as plt
import numpy as np


pods.notebook.display_google_book(id='spcAAAAAMAAJ', page=72)

data = pods.datasets.olympic_marathon_men()
x = data['X']
y = data['Y']

plt.plot(x, y, 'rx')
plt.xlabel('year')
plt.ylabel('pace in min/km')
plt.savefig('./initial_fig.png')

# next we set the linear algebra error function
#
def error_function():
    m = -0.4
    c = 80
    c = (y - m * x).mean()

    # use the equation to calculate the value of m
    m = ((y - c) * x).sum() / (x ** 2).sum()
    # list all the year using the linspace function
    x_test = np.linspace(1890, 2020, 130)[:, None]
    f_test = m * x_test + c

    # de
    for i in np.arange(10):
        m = ((y - c) * x).sum() / (x * x).sum()
        c = (y - m * x).sum() / y.shape[0]
    print(m)
    print(c)



