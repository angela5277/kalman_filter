#set up virtual enviroment  <br />
python3.7 -m venv venv  <br />
#activate virtual env  <br />
source venv/bin/activate  <br />
#install requirements  <br />
pip install -r requirements.txt

#Description
kalman filter contains the predict function and 
the MLE estimates of parameters<br />
data_processing handles option data filtering <br />
option_pricer calculate the blackschores <br />
backtest_portfolios contains four option portfolio <br/>
demo.py is the test case