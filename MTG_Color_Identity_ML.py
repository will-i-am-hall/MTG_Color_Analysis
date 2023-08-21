# -*- coding: utf-8 -*-
"""
MTG Color Identity ML Project

Created on Thu Jul 27 20:54:01 2023

@author: Will Hall
"""
import pandas as pd
from numpy import array, subtract
from numpy import polyfit as pf
import matplotlib.pyplot as plt
from sklearn import tree 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

df = pd.read_csv('MTG_Color_Identity.csv')

colors = df.iloc[:, 1:].values
white = df.loc[:, 'White'].values
blue = df.loc[:, 'Blue'].values
black = df.loc[:, 'Black'].values
red = df.loc[:, 'Red'].values
green = df.loc[:, 'Green'].values

dates = df[['DATE']]
dates = dates.apply(pd.to_datetime)


#for comparison
wmean = white.mean()
umean = blue.mean()
bmean = black.mean()
rmean = red.mean()
gmean = green.mean()

means = array([[wmean, umean, bmean, rmean, gmean]])


###############################################################################
###########################  Machine Learning  ################################
###############################################################################


########### All Colors ########################################################
print('----------------------- All Colors ----------------------------------')

example_date = pd.DataFrame(['4/20/2024'])
example_date = example_date.apply(pd.to_datetime)

x_train, x_test, y_train, y_test = train_test_split(dates, colors, train_size = 0.99)

print('~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Decision Trees')
print('~~~~~~~~~~~~~~~~~~~~~~~~~\n')

reg = tree.DecisionTreeRegressor()
reg = reg.fit(x_train, y_train)
dtpredictions = reg.predict(example_date)
dt_pred = reg.predict(x_test)
print('Example prediction      : ', dtpredictions)
print('Example diff from mean  :', subtract(means, dtpredictions))
print('Mean Squared Error      : ', mean_squared_error(y_test, dt_pred))

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print('Support Vector Machines')
print('~~~~~~~~~~~~~~~~~~~~~~~~\n')

sregw = SVR().fit(x_train, y_train[:,0])
sregu = SVR().fit(x_train, y_train[:,1])
sregb = SVR().fit(x_train, y_train[:,2])
sregr = SVR().fit(x_train, y_train[:,3])
sregg = SVR().fit(x_train, y_train[:,4])

svrpw = sregw.predict(x_test)
svrpu = sregu.predict(x_test)
svrpb = sregb.predict(x_test)
svrpr = sregr.predict(x_test)
svrpg = sregg.predict(x_test)

svrpxw = sregw.predict(example_date)
svrpxu = sregu.predict(example_date)
svrpxb = sregb.predict(example_date)
svrpxr = sregr.predict(example_date)
svrpxg = sregg.predict(example_date)

svrp = array(list(zip(svrpw, svrpu, svrpb, svrpr, svrpg)))

svrpredictions = array(list(zip(svrpxw, svrpxu, svrpxb, svrpxr, svrpxg)))
print('Example prediction      : ', svrpredictions)
print('Example diff from mean  :', subtract(means, svrpredictions))
print('Mean Squared Error      :', mean_squared_error(y_test, svrp))
print('----------------------------------------------------------------------')


########### By Color w/ Graphs ################################################

#11111111111111111111111    Decision Trees    111111111111111111111111111111111
########### White #############################################################

wx_train, wx_test, wy_train, wy_test = train_test_split(dates, white, train_size = 0.66)

wreg = tree.DecisionTreeRegressor()
wreg = wreg.fit(wx_train, wy_train)

dtpredw = wreg.predict(wx_test)
wtrend = pf(wy_test, dtpredw, 1)
wco, woff = wtrend
wline = wco * wy_test + woff

########### Blue  #############################################################

ux_train, ux_test, uy_train, uy_test = train_test_split(dates, blue, train_size = 0.66)

ureg = tree.DecisionTreeRegressor()
ureg = ureg.fit(ux_train, uy_train)

dtpredu = ureg.predict(ux_test)
utrend = pf(uy_test, dtpredu, 1)
uco, uoff = utrend
uline = uco * uy_test + uoff


########### Black #############################################################

bx_train, bx_test, by_train, by_test = train_test_split(dates, black, train_size = 0.66)

breg = tree.DecisionTreeRegressor()
breg = breg.fit(bx_train, by_train)

dtpredb = breg.predict(bx_test)
btrend = pf(by_test, dtpredb, 1)
bco, boff = btrend
bline = bco * by_test + boff

########### Red   #############################################################

rx_train, rx_test, ry_train, ry_test = train_test_split(dates, red, train_size = 0.66)

rreg = tree.DecisionTreeRegressor()
rreg = rreg.fit(rx_train, ry_train)

dtpredr = rreg.predict(rx_test)
rtrend = pf(ry_test, dtpredr, 1)
rco, roff = rtrend
rline = rco * ry_test + roff

########### Green #############################################################

gx_train, gx_test, gy_train, gy_test = train_test_split(dates, green, train_size = 0.66)

greg = tree.DecisionTreeRegressor()
greg = greg.fit(gx_train, gy_train)

dtpredg = greg.predict(gx_test)
gtrend = pf(gy_test, dtpredg, 1)
gco, goff = gtrend
gline = gco * gy_test + goff

###################            Plotting             ###########################

fig, axs = plt.subplots(2,3)
fig.suptitle('MTG Color Analysis: Decision Trees')
fig.tight_layout()

axs[0,0].set_title('White, MSE: ' + str(mean_squared_error(wy_test, dtpredw)))
axs[0,0].set_ylabel('Predicted Values')
axs[0,0].set_xlabel('Real Values')
axs[0,0].scatter(wy_test, dtpredw, c = 'y')
axs[0,0].plot(wy_test, wline, '#444444')
axs[0,0].grid(True)

axs[0,1].set_title('Blue, MSE: ' + str(mean_squared_error(uy_test, dtpredu)))
axs[0,1].set_ylabel('Predicted Values')
axs[0,1].set_xlabel('Real Values')
axs[0,1].scatter(uy_test, dtpredu, c = 'b')
axs[0,1].plot(uy_test, uline, '#444444')
axs[0,1].grid(True)

axs[0,2].set_title('Black, MSE: ' + str(mean_squared_error(by_test, dtpredb)))
axs[0,2].set_ylabel('Predicted Values')
axs[0,2].set_xlabel('Real Values')
axs[0,2].scatter(by_test, dtpredb, c = 'm')
axs[0,2].plot(by_test, bline, '#444444')
axs[0,2].grid(True)

axs[1,0].set_title('Red, MSE: ' + str(mean_squared_error(ry_test, dtpredr)))
axs[1,0].set_ylabel('Predicted Values')
axs[1,0].set_xlabel('Real Values')
axs[1,0].scatter(ry_test, dtpredr, c = 'r')
axs[1,0].plot(ry_test, rline, '#444444')
axs[1,0].grid(True)

axs[1,2].set_title('Green, MSE: ' + str(mean_squared_error(gy_test, dtpredg)))
axs[1,2].set_ylabel('Predicted Values')
axs[1,2].set_xlabel('Real Values')
axs[1,2].scatter(gy_test, dtpredg, c = 'g')
axs[1,2].plot(gy_test, gline, '#444444')
axs[1,2].grid(True)

fig.delaxes(axs[1,1])

#2222222222222222222222     Nearest Neighbors    222222222222222222222222222222
########### White #############################################################

wx_train2, wx_test2, wy_train2, wy_test2 = train_test_split(dates, white, train_size = 0.66)

wreg2 = SVR()
wreg2 = wreg2.fit(wx_train2, wy_train2)

svrpredw = wreg2.predict(wx_test2)
wtrend2 = pf(wy_test2, svrpredw, 1)
wco2, woff2 = wtrend2
wline2 = wco2 * wy_test2 + woff2

########### Blue  #############################################################

ux_train2, ux_test2, uy_train2, uy_test2 = train_test_split(dates, blue, train_size = 0.66)

ureg2 = SVR()
ureg2 = ureg2.fit(ux_train2, uy_train2)

svrpredu = ureg2.predict(ux_test2)
utrend2 = pf(uy_test2, svrpredu, 1)
uco2, uoff2 = utrend2
uline2 = uco2 * uy_test2 + uoff2


########### Black #############################################################

bx_train2, bx_test2, by_train2, by_test2 = train_test_split(dates, black, train_size = 0.66)

breg2 = SVR()
breg2 = breg2.fit(bx_train2, by_train2)

svrpredb = breg2.predict(bx_test2)
btrend2 = pf(by_test2, svrpredb, 1)
bco2, boff2 = btrend2
bline2 = bco2 * by_test2 + boff2

########### Red   #############################################################

rx_train2, rx_test2, ry_train2, ry_test2 = train_test_split(dates, red, train_size = 0.66)

rreg2 = SVR()
rreg2 = rreg2.fit(rx_train2, ry_train2)

svrpredr = rreg2.predict(rx_test2)
rtrend2 = pf(ry_test2, svrpredr, 1)
rco2, roff2 = rtrend2
rline2 = rco2 * ry_test2 + roff2

########### Green #############################################################

gx_train2, gx_test2, gy_train2, gy_test2 = train_test_split(dates, green, train_size = 0.66)

greg2 = SVR()
greg2 = greg2.fit(gx_train2, gy_train2)

svrpredg = greg2.predict(gx_test2)
gtrend2 = pf(gy_test2, svrpredg, 1)
gco2, goff2 = gtrend
gline2 = gco2 * gy_test2 + goff2

###################            Plotting             ###########################

fig2, axs2 = plt.subplots(2,3)
fig2.suptitle('MTG Color Analysis: Support Vector Machine')
fig2.tight_layout()

axs2[0,0].set_title('White, MSE: ' + str(mean_squared_error(wy_test2, svrpredw)))
axs2[0,0].set_ylabel('Predicted Values')
axs2[0,0].set_xlabel('Real Values')
axs2[0,0].scatter(wy_test2, svrpredw, c = 'y')
axs2[0,0].plot(wy_test2, wline2, '#444444')
axs2[0,0].grid(True)

axs2[0,1].set_title('Blue, MSE: ' + str(mean_squared_error(uy_test2, svrpredu)))
axs2[0,1].set_ylabel('Predicted Values')
axs2[0,1].set_xlabel('Real Values')
axs2[0,1].scatter(uy_test2, svrpredu, c = 'b')
axs2[0,1].plot(uy_test2, uline2, '#444444')
axs2[0,1].grid(True)

axs2[0,2].set_title('Black, MSE: ' + str(mean_squared_error(by_test2, svrpredb)))
axs2[0,2].set_ylabel('Predicted Values')
axs2[0,2].set_xlabel('Real Values')
axs2[0,2].scatter(by_test2, svrpredb, c = 'm')
axs2[0,2].plot(by_test2, bline2, '#444444')
axs2[0,2].grid(True)

axs2[1,0].set_title('Red, MSE: ' + str(mean_squared_error(ry_test2, svrpredr)))
axs2[1,0].set_ylabel('Predicted Values')
axs2[1,0].set_xlabel('Real Values')
axs2[1,0].scatter(ry_test2, svrpredr, c = 'r')
axs2[1,0].plot(ry_test2, rline2, '#444444')
axs2[1,0].grid(True)

axs2[1,2].set_title('Green, MSE: ' + str(mean_squared_error(gy_test2, svrpredg)))
axs2[1,2].set_ylabel('Predicted Values')
axs2[1,2].set_xlabel('Real Values')
axs2[1,2].scatter(gy_test2, svrpredg, c = 'g')
axs2[1,2].plot(gy_test2, gline2, '#444444')
axs2[1,2].grid(True)

fig2.delaxes(axs2[1,1])
plt.show()

