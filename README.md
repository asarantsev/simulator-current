# simulator-current
Web site for the simulator: https://asarantsev.pythonanywhere.com/

UPDATE: June 20, uploaded simKDE.py which simulates NSIMS * nYears data of multivariate innovations. Added innovations.xlsx for original values of innovations and filled.xlsx for imputed values, so all five series in filled.xlsx have size 97. This imputation is done using linear regression with uniformly sampled regression residuals, see innovations.py. This Python code file reads innovations.xlsx and writes filled.xlsx. 

UPDATE: June 20, corrected a misprint in flask_app7.py

UPDATE: June 18, https://my-finance.org/2025/06/18/updated-simulator-for-rate-and-volatility/

flask_app7.py BACKEND

main_page7.html FRONTEND (home page)

easy_page.html FRONTEND (simplified version page)

response_page.html FRONTEND (output page)

overall.xlsx MAIN DATA FILE

validation.py FITTING MODEL

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

UPDATE: June 11, https://my-finance.org/2025/06/10/new-simulator/

flask_app6.py BACKEND

main_page5.html FRONTEND (home page)

response_page.html FRONTEND (output page)

overall.xlsx MAIN DATA FILE

validation.py FITTING MODEL

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Old version of the simulator, May 30

flask_app.py BACKEND

main_page.html LANDING PAGE

response_page.html FRONTEND

annual.xlsx MAIN DATA FILE

international.xlsx ANOTHER DATA FILE

Updated, June 10: 
flask-appK.py and main_pageK.html for the same K are new versions
