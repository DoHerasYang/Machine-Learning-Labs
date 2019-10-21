import pods
import pylab as plt
import numpy as np
import random


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

    plt.plot(x_test, f_test, 'b-')
    plt.plot(x, y, 'rx')

    for i in np.arange(10):
        m = ((y - c) * x).sum() / (x * x).sum()
        c = (y - m * x).sum() / y.shape[0]
    print(m)
    print(c)

    m = ((y - c) * x).sum() / (x ** 2).sum()
    print(m)

    f_test = m * x_test + c
    plt.plot(x_test, f_test, 'b-')
    plt.plot(x, y, 'rx')

# TODO:There is a problem here, we seem to need many interations to get to a good solution. Let's explore what's going
#  on. Write code which alternates between updates of c and m. Include the following features in your code.
#   (a) Initialise with m=-0.4 and c=80.
#   (b) Every 10 iterations compute the value of the objective function for the training data and print it to the screen
#   (you'll find hints on this in the lab from last week.
#   (c) Cause the code to stop running when the error change over less than 10 iterations is smaller than  1×10−4 .
#   This is known as a stopping criterion.
#   Why do we need so many iterations to get to the solution?


def answer_Q1():
    m = -0.4
    c = 80
    iteration_num = 0
    value_diff = random.randint(1, 10)
    obj_function = m * x + c
    # this is the objective function for reduce the quare difference
    objdiff_function = ((y - obj_function) ** 2).sum()

    while value_diff > 0.0001:
        temp_value = objdiff_function
        # update the value of m and c
        m = ((y - c) * x).sum() / (x * x).sum()
        c = (y - m * x).sum() / y.shape[0]
        obj_function = m * x + c

        if iteration_num % 10 == 0:
            objdiff_function = ((y - obj_function) ** 2).sum()
            print("Now %d Iteration" % (iteration_num), "Objective:", (objdiff_function))
            value_diff = temp_value - objdiff_function
        iteration_num += 1



if __name__ == '__main__':
    error_function()



