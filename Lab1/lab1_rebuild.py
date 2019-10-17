import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# Get the data from the csv file which have been downloaded from the
film_deaths = pd.read_csv('./film-death-counts-Python.csv')
# The return value is all the data from the file and the type of film_deaths is <class 'pandas.core.frame.DataFrame'>


# Summarize the data
film_deaths.describe()

# Display the data to screen
# Use the  DataFrame['row_name'] to select the all the row
print(film_deaths['Year'])
print(film_deaths['Body_Count'])

# Plot the data
plt.figure()  # We only need one plt.figure(), if we set more than one, the below figures will be blank.
plt.plot(film_deaths['Year'], film_deaths['Body_Count'], 'rx')
plt.savefig("./basic.png")
plt.show()

# pylab.plot()
# **Markers**
#
# =============    ===============================
# character        description
# =============    ===============================
# ``'.'``          point marker
# ``','``          pixel marker
# ``'o'``          circle marker
# ``'v'``          triangle_down marker
# ``'^'``          triangle_up marker
# ``'<'``          triangle_left marker
# ``'>'``          triangle_right marker
# ``'1'``          tri_down marker
# ``'2'``          tri_up marker
# ``'3'``          tri_left marker
# ``'4'``          tri_right marker
# ``'s'``          square marker
# ``'p'``          pentagon marker
# ``'*'``          star marker
# ``'h'``          hexagon1 marker
# ``'H'``          hexagon2 marker
# ``'+'``          plus marker
# ``'x'``          x marker
# ``'D'``          diamond marker
# ``'d'``          thin_diamond marker
# ``'|'``          vline marker
# ``'_'``          hline marker
# =============    ===============================
#
# **Line Styles**
#
# =============    ===============================
# character        description
# =============    ===============================
# ``'-'``          solid line style
# ``'--'``         dashed line style
# ``'-.'``         dash-dot line style
# ``':'``          dotted line style
# =============    ===============================
#
# Example format strings::
#
#     'b'    # blue markers with default shape
#     'or'   # red circles
#     '-g'   # green solid line
#     '--'   # dashed line with default color
#     '^k:'  # black triangle_up markers connected by a dotted line
#
# **Colors**
#
# The supported color abbreviations are the single letter codes
#
# =============    ===============================
# character        color
# =============    ===============================
# ``'b'``          blue
# ``'g'``          green
# ``'r'``          red
# ``'c'``          cyan
# ``'m'``          magenta
# ``'y'``          yellow
# ``'k'``          black
# ``'w'``          white
# =============    ===============================
#
# and the ``'CN'`` colors that index into the default property cycle.
#
# If the color is the only part of the format string, you can
# additionally use any  `matplotlib.colors` spec, e.g. full names
# (``'green'``) or hex strings (``'#008000'``).

# Identify the specific range of the data
# use the film_deaths['Body_Count'] > 200 to set the index to classify the data and return value is bool Panda Series
#
print(film_deaths[film_deaths['Body_Count'] > 200].sort_values('Body_Count', ascending=False))
# redefine the variable

# re-plot the graph by histograming the data
# The function variable.hist() is to show the frequency of row values

film_deaths['Body_Count'].hist(bins=20)
plt.title('Histogram of Film Kill count')
plt.savefig('./histogram.png')
plt.show()

# re-plot the graph by red Mark_X and plot the logarithm of the counts
plt.plot(film_deaths['Year'], film_deaths['Body_Count'], 'rx')
ax = plt.gca()  # give a handle to the current axis / Or I think it finds the Axes and make variable direct to the axes
ax.set_yscale('log')  # set the logarithmic death scale
plt.title('Film Death against Year')
plt.xlabel('year')
plt.ylabel('deaths')
plt.savefig('./logarithmic.png')
plt.show()

# Next, we need the python to calculate approximate probabilities of deaths of film which
# from the movie body count website has over 40 deaths
deaths = (film_deaths.Body_Count > 40).sum()  # sum() to return the sum of Dataframe
total_films = film_deaths.Body_Count.count()  # only the Series can return a number
prob_death = float(deaths)/float(total_films)
print("Probability of deaths being greather than 40 is:", prob_death)

# Next, We will do the conditioning probability of death for each movie which relies on the year
for year in [2000, 2002]:
    deaths1 = (film_deaths.Body_Count[film_deaths.Year == year] > 40).sum()
    total_films1 = (film_deaths.Year == year).sum()
    prob_betch1 = float(deaths1) / float(total_films1)
    print("Probability of deaths being greather than 40 in year", year, "is:", prob_betch1)

# TODO: Compute the probability for the number of deaths being over 40 for each year we have in
#  film_deaths data frame. Store the result in a numpy array and plot the probabilities against
#  the years using the plot command from matplotlib.

# Train	of thought: The hinge to solve this problem is to find the range of year from the dataset. In Panda, I use the
# the drop_duplicates function to remove the repeated row which has the same year. For creating the new array by Numpy,
# I think It would be better to use the list to creat array to avoid the error. As we know, the type of column of Year
# in death_films is int we can use the int variable in If function.
#
year_range = film_deaths.drop_duplicates('Year').sort_values('Year', ascending = True)
year_list = []
prob_list = []
for year in year_range.Year:
    deaths2 = (film_deaths.Body_Count[film_deaths.Year == year] > 40).sum()
    total_films2 = (film_deaths.Year == year).sum()
    prob_betch2 = float(deaths2) / float(total_films2)
    year_list.append(year)
    prob_list.append(prob_betch2)
year_axis = np.array(year_list)
prob_ayis = np.array(prob_list)

plt.plot(year_axis,prob_ayis, 'rx')
plt.savefig('./everyyear.png')

# Rules of Probability
# The scatter plot of deaths vs year that we created above can be seen as a joint probability distribution
# We represent a joint probability using the notation.
# Question: What's the probability that the number of deaths was over 40 and the year was 2002?
#

temp_year = 2000
deaths = (film_deaths.Body_Count[film_deaths.Year == temp_year] > 40).sum()
total_films = film_deaths.Body_Count.count()
prob_death = float(deaths) / float(total_films)
print("Probability of deaths being greather than 40 and year being", temp_year, "is:", prob_death)


# The Product Rule
# for p(Y = y, T = t) = p(Y = y|T=t)p(T = t)
# we set the t=2000
# film_deaths.Body_Count.count() -> The total number of sample
# film_deaths.Year == 2002).sum()-> The total number of sample im 2002
# film_deaths.Body_Count[film_deaths.Year == 2002] > 40).sum() -> The number of Body_Count is larger than the 40

p_t = float((film_deaths.Year == 2002).sum()) / float(film_deaths.Body_Count.count())  # sum() to calculate the number of the movies which have the same year
p_y_given_t = float((film_deaths.Body_Count[film_deaths.Year == 2002] > 40).sum()) / float((film_deaths.Year == 2002).sum())
p_y_and_t = float((film_deaths.Body_Count[film_deaths.Year == 2002] > 40).sum()) / float(film_deaths.Body_Count.count())
print("P(t) is", p_t)
print("P(y|t) is", p_y_given_t)
print("P(y,t) is", p_y_and_t)

# The Sum Rule
# TODO: Now, please add all the p(y,t) to get the result of P(Y = y)
#
year_range = film_deaths.drop_duplicates('Year').sort_values('Year', ascending = True)  # reuse the range of year
p_y = float(0)  # set the initial value of p_y, attention: we need the float value.
for year in year_range.Year:
    p_t = float((film_deaths.Year == year).sum()) / float(film_deaths.Body_Count.count())  # sum() to calculate the number of the movies which have the same year
    p_y_given_t = float((film_deaths.Body_Count[film_deaths.Year == year] > 40).sum()) / float((film_deaths.Year == 2002).sum())
    p_y_and_t = float((film_deaths.Body_Count[film_deaths.Year == year] > 40).sum()) / float(film_deaths.Body_Count.count())
    p_y = float(p_y + p_y_and_t)
print("P(y):", p_y)


