# ml-for-survival
This repository contains the code and models of the paper **"Machine Learning Models for the Prediction of Breast Cancer Prognostic: Application and Comparison Based on a Retrospective Cohort Study"**

**You can use this repository to**
1. build your own predictive models based on your data(mlmain.py cindexandroc.R kmcurves.py) (run the programs according to this order)
2. use our models to get the risk score of the breast cancer patients (models.py)
3. use your data with independent variables and survival outcomes to verify our models (models.py)

**Explanation of each file**

**code:**
- mlmain.py: the mean program for data processing and build predictive models
- cindexandroc.R: evaluate the C-index and time dependent AUC based on the data outputted by mlmain.py
- kmcurves.py: obtain the figure of survival curves based on the data outputted by cindexandroc.R
- models.py: load the models we packed and use the models to predict the risk score.

**data templates:**
- df.xlsx: data used to build predictive models
- coxentest.xlsx: data used to test the cox-en model
- coxtest.xlsx: data used to test the cox model
- svmrsftest.xlsx: data used to test the svm and rsf model
- rsfimportance.xlsx:  data obtained from iris-importance.htm (users are supposed to move the data from iris-importance.htm to rsfimportance.xlsx)

**packed models:**
- cox.pkl
- cox-en.pkl
- svm.pkl
- rsf.pkl

**Following are some tips help you to use our code and data templates:**
1. Please move all the files and codes to the path"C:\code"
2. The details of the code of Cox-EN, SVM and RSF are in https://scikit-survival.readthedocs.io/en/latest/index.html
3. The datasets in the repository are meaningless and randomly generated to be used only as templates. Please input the data you want to analyze in our templates.
4. csv and xlsx are both used and please make sure you are using the right format
5. How the variable was defined is in variables_definition.doc
