import ntpath
import wx
import traceback
import sys
import pandas as pd
from pandas import ExcelWriter
from sklearn import preprocessing

def show_error():
    message = ''.join(traceback.format_exception(*sys.exc_info()))
    dialog = wx.MessageDialog(None, message, 'Error!', wx.OK | wx.ICON_ERROR)
    dialog.ShowModal()
    
def onUploadButton(event):
    openFileDialog = wx.FileDialog(frame, "Open", "", "",
                                   "Excel files (*.xlsx)|*.xlsx",
                                   wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    openFileDialog.ShowModal()
    global filePath
    filePath = openFileDialog.GetPath()
    fileName = ntpath.basename(filePath)
    textField.SetValue("%s" % (fileName))
    textField_1.SetValue("%s" % (filePath))

    global data
    data = pd.read_excel(filePath)
    data = data.dropna()
    
    global data_columns
    data_columns = pd.read_excel(filePath, skipfooter = data.size - 1)
    
    #Describe
    global describe
    describe_f = data.describe(include = ['object', 'float', 'int'])
    describe_l = data.describe()
    describe = pd.concat([describe_f, describe_l], sort = False)
    
    combo_feature_label = Feature_Label_Selection()
    combo_feature_label.ComboBox()
    
    problem_select = Problem_Selection()
    problem_select.problem_select()
    
class Feature_Label_Selection():
    def __init__(self):
        pass 
    
    def ComboBox(self):
        global Select_Features
        Select_Features = wx.ComboBox(panel, wx.ID_ANY, 'Select Features >>', 
                              choices = data_columns.columns, 
                              pos = (180, 170), size = (140, 60), style=wx.LB_MULTIPLE)
    
   
        global Select_Labels
        Select_Labels = wx.ComboBox(panel, wx.ID_ANY, 'Select Labels >>', 
                                choices = data_columns.columns,
                                pos = (360, 170), size = (140, 60), style=wx.LB_MULTIPLE)
        
        global describe_data
        describe_data = wx.ComboBox(panel, wx.ID_ANY, 'Describe data >>', 
                                choices = data_columns.columns,
                                pos = (10, 130), size = (140, 60))
    
        global feature_list
        feature_list = []
    
        global label_list
        label_list = []
    
        Select_Features.Bind(wx.EVT_COMBOBOX, self.OnSelectFeatures)
        Select_Labels.Bind(wx.EVT_COMBOBOX, self.OnSelectLabels)
        describe_data.Bind(wx.EVT_COMBOBOX, self.OnDescribe_data)
  
    def OnSelectFeatures(self,event):
        textField.Show()
        textField_1.Show()
        textField_2.Hide()
        feature_list.append(Select_Features.GetValue())
        feature_l = ', '.join(str(x) for x in feature_list)
        textField.SetValue(feature_l)
    
    def OnSelectLabels(self,event):
        textField.Show()
        textField_1.Show()
        textField_2.Hide()
        label_list.append(Select_Labels.GetValue())
        label_l = ', '.join(str(x) for x in label_list)
        textField_1.SetValue(label_l)
        
    def OnDescribe_data(self, event):
        textField.Hide()
        textField_1.Hide()   
        textField_2.Show()
    
        textField_2.AppendText('\n\n'+describe_data.GetValue() + ' Details  >>'+'\n')
        textField_2.AppendText(describe[describe_data.GetValue()].fillna('--').to_string())
        
class Problem_Selection():
    def __init__(self):
        pass
  
    def problem_select(self):
        problem = ['Regression', 'Classification', 'Neural Network']
        global Select_Problem
        Select_Problem = wx.ComboBox(panel, wx.ID_ANY, 'Select Problem >>', 
                              choices = problem, 
                              pos = (10, 170), size = (140, 60), style=wx.LB_MULTIPLE)
        
        regression_models = ['Linear Regression', 'Support Vector Machines', 
                                 'Decision Tree', 'Random Forest']
        
        global Select_regression_Model              
        Select_regression_Model = wx.ComboBox(panel, wx.ID_ANY, 'Select Model >>', 
                              choices = regression_models, 
                              pos = (10, 210), size = (140, 60), style=wx.LB_MULTIPLE)
        
        global Select_class_Model
        classification_models = ['Logistic Regression','K-Nearest Neighbor', 
                                     'Naive Bayes', 'XGBoost', 'Random Forest', 'LightGBM']
        
        Select_class_Model = wx.ComboBox(panel, wx.ID_ANY, 'Select Model >>', 
                              choices = classification_models, 
                              pos = (10, 210), size = (140, 60), style=wx.LB_MULTIPLE)
        
        
        x = DecisionTree()
        x.layout_DecisionTree()
    
        y = RandomForest()
        y.layout_RandomForest()
                    
        Select_Problem.Bind(wx.EVT_COMBOBOX, self.onProblem_Selection)
        Select_regression_Model.Bind(wx.EVT_COMBOBOX, lambda event: OnSelectRegressionModel(event, x , y))
        Select_class_Model.Bind(wx.EVT_COMBOBOX, OnSelectClassModel)
        
        #Neural Part - wrote latar, So comment
        global Select_Neural_Network
        Neural_Network_models = ['Sequential', 'Model']
        
        Select_Neural_Network = wx.ComboBox(panel, wx.ID_ANY, 'Select Model >>', 
                              choices = Neural_Network_models, 
                              pos = (10, 210), size = (140, 60), style=wx.LB_MULTIPLE)
        
        z = Keras_NeuralNetwork()
        z.layout_Keras_NeuralNetwork_compile()
        z.layout_Keras_NeuralNetwork_layer()
        #y = Keras_NeuralNetwork()
        Select_Neural_Network.Bind(wx.EVT_COMBOBOX, lambda event: OnSelectNeuralNetwork(event, z))
        
        Lock_button.Bind(wx.EVT_BUTTON, lambda event: On_Lock_button(event, z))
        
        Add_button.Bind(wx.EVT_BUTTON, lambda event: On_Add_button(event, z))
        
        #For Neural only, need something to store layer details
        global lst_layer, lst_size, lst_Input_dim, lst_activation 
        lst_layer = []
        lst_size = []
        lst_Input_dim = []
        lst_activation = []
        
              
    def onProblem_Selection(self, event):
        if(Select_Problem.GetValue() == 'Regression'):
            Select_class_Model.Hide()
            Select_Neural_Network.Hide()
            Select_regression_Model.Show()
            
        if(Select_Problem.GetValue() == 'Classification'):   
            Select_regression_Model.Hide()
            Select_Neural_Network.Hide()
            Select_class_Model.Show()
            
        if(Select_Problem.GetValue() == 'Neural Network'):
            Select_class_Model.Hide()
            Select_regression_Model.Hide()
            Select_Neural_Network.Show()
            
       
##############################################################################

def OnSelectRegressionModel(event,x,y):
    print('model regression event')
    coeff_button.Hide()
    
    upload_Input_button.Show()
    download_results_button.Show()
    #global model
    
    if(Select_regression_Model.GetValue() == 'Linear Regression'):
        Linear.Linear_Regression()
        coeff_button.Show()

        
    if(Select_regression_Model.GetValue() == 'Support Vector MAchines'):
        #Support_Vector_Machines()
        print('Support Vector Machine')
        
        
    if(Select_regression_Model.GetValue() == 'Decision Tree'):
        #y.Output_Results()
        y.Select_n_estimators.Hide()
        y.Select_max_features_decision.Hide()
        y.Select_maxdepth_decision.Hide()
        y.Select_random_state_decision.Hide()
        
        x.Select_maxdepth.Show()
        x.Select_criterion.Show()
        x.Select_min_samples_leaf.Show()
        x.Select_random_state.Show()

      
    if(Select_regression_Model.GetValue() == 'Random Forest'):
        print('a')
        #y.Output_Results()
        x.Select_criterion.Hide()
        x.Select_maxdepth.Hide()
        x.Select_min_samples_leaf.Hide()
        x.Select_random_state.Hide()
        
        y.Select_n_estimators.Show()
        y.Select_max_features_decision.Show()
        y.Select_maxdepth_decision.Show()
        y.Select_random_state_decision.Show()
        

def OnSelectClassModel(event):
    print('OnSelectClassModel')
    if(Select_class_Model.GetValue() == 'Logistic Regression'):
        Logistic_Regression.Logistic__Regression()
    if(Select_class_Model.GetValue() == 'K-Nearest Neighbor'):
        KNearestNeighbour.KNearest_Neighbour()
    if(Select_class_Model.GetValue() == 'Naive Bayes'):
        Naivebayes.Naive_bayes()
    if(Select_class_Model.GetValue() == 'XGBoost'):
        Xgboost.XGBoost()
    if(Select_class_Model.GetValue() == 'Random Forest'):
        RandomForestClassifier.RandomForest()
    if(Select_class_Model.GetValue() == 'LightGBM'):
        LightGBM.Light_GBM()
        
def OnSelectNeuralNetwork(event, z):
    print('OnSelectNeuralNetwork')
    
    wx.StaticText(panel, -1, "Add Layers, and then Lock it         Total Layers -- > ", pos=(168, 214))
    
    z.Select_optimizer.Hide()
    z.Select_loss.Hide()
    z.Select_metrics.Hide()
    
    Add_button.Show()
    Lock_button.Show()    
    textField_size.Show()
    textField_Input_dim.Show()
    textField_total_layers.Show()    
    z.Select_layer.Show()
    z.Select_activation.Show()
    

 
###############################################################################
def onCoeffButton(event):
    textField.Hide()
    textField_1.Hide()
    textField_2.Show()
    
    textField_2.AppendText('\n\n'+ Select_regression_Model.GetValue() + ' Coefficients' + '\n')
    
    for index, row in coeff_linear.iterrows():
        textField_2.AppendText('      ->' + row['Features'] + '      ' + str(row['Coeff']))
        textField_2.AppendText('\n')
        
def On_Add_button(event, z):
    lst_layer.append(z.layers)
    lst_size.append(textField_size.GetValue())
    lst_Input_dim.append(textField_Input_dim.GetValue())
    lst_activation.append(z.activation)
        
    textField.Hide()
    textField_1.Hide()       
    textField_2.Show()       
    
    textField_2.AppendText('\n******************************************')
        
    textField_2.AppendText('\nNeural Network Layer Added >>')
    textField_2.AppendText('\n\n   Layer: ['+z.layers+']')
    textField_2.AppendText('    Size: ['+textField_size.GetValue()+']')
    textField_2.AppendText('    Input_dim: ['+textField_Input_dim.GetValue()+']')
    textField_2.AppendText('    Activation: ['+z.activation+']')
    
        
def On_Lock_button(event, y):
    
    wx.StaticText(panel, -1, "Select options to compile your Neural Network   -- > ", pos=(168, 214))
    
    #y.layout_Keras_NeuralNetwork_layer()
    Add_button.Hide()
    Lock_button.Hide()    
    textField_size.Hide()
    textField_Input_dim.Hide()
    textField_total_layers.Hide()  
    y.Select_layer.Hide()
    y.Select_activation.Hide()
    
    #y.layout_Keras_NeuralNetwork_compile()
    y.Select_optimizer.Show()
    y.Select_loss.Show()
    y.Select_metrics.Show()
    
    upload_Input_button.Show()
    download_results_button.Show()
    
    textField_2.AppendText('\n\n Layers are Locked now --> Prepare to compile it all   ')
        
def On_upload_Input_button(event):
    openFileDialog = wx.FileDialog(frame, "Open", "", "",
                                   "Excel files (*.xlsx)|*.xlsx",
                                   wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    openFileDialog.ShowModal()
    
    global filePath_input_loc
    global filePath_input_notes
    global filePath_input
    filePath_input = openFileDialog.GetPath()
    print(filePath_input)
    
    
    global data_input
    data_input = pd.read_excel(filePath_input)
    data_input = data_input[feature_list]
    print(data_input.info())
    
    filePath_input_loc = filePath_input.replace('.','_output.')
    
def On_download_results_button(event):
    print('In On_download_results_button')
    if(Select_regression_Model.GetValue() == 'Linear Regression'):
        print('Linear Regression')
        Linear.Linear_Regression()
        #Linear.Output_Results()

    if(Select_regression_Model.GetValue() == 'Support Vector MAchines'):
        print('Support Vector MAchines')

    if(Select_regression_Model.GetValue() == 'Decision Tree'):
        Decision = DecisionTree()
        Decision.Decision_Tree()
      
    if(Select_regression_Model.GetValue() == 'Random Forest'):
        Random = RandomForest()
        Random.Random_Forest()

    
    resultText = ('Output file downloaded : '+filePath_input_loc)
    wx.MessageBox(message=resultText,
                          caption='Download Complete',
                          style=wx.OK | wx.ICON_INFORMATION)
    
        
def On_download_notes_button(event):
    notes = textField_2.GetValue()
    
       #
    global filePath_input_notes
    filePath_input_notes = filePath_input.replace('.xlsx', 'Notes.txt')
    print(filePath_input_notes)
    
    with open(filePath_input_notes, 'w') as fobj:
        fobj.write(notes)
        
    resultText = ('Notes downloaded : '+filePath_input_notes)
    wx.MessageBox(message=resultText,
                          caption='Download Complete',
                          style=wx.OK | wx.ICON_INFORMATION)       
    
###############################################################################        

class Linear():
    def __init__():
        pass
    
    def Linear_Regression():
      try:  
        from sklearn import linear_model
        from sklearn.model_selection import train_test_split
        from matplotlib import pyplot as plt
        import numpy as np
        
        global linear_le
        linear_le = preprocessing.LabelEncoder()
        
        data_encode = data

        for column in data_encode.columns:
            if data_encode[column].dtype == type(object):
                data_encode[column] = linear_le.fit_transform(data_encode[column].astype(str))
    
        df = data[feature_list]
        y = data[label_list]
        
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test) 
        
        #Here Trying to implement model
        import statsmodels.api as sm
        logit_model = sm.OLS(y_train, X_train)
        result = logit_model.fit()
        #print(result.summary2())
        summary = str(result.summary2())
        
        #fitting data into Linear Model
        #global linear_model
        linear_model = linear_model.LinearRegression()
        linear_model = linear_model.fit(X_train, y_train)
        predict = linear_model.predict(X_test)
        
        #coeff
        global coeff_linear
        coeff_linear = pd.concat([pd.DataFrame(df.columns, columns = ['Features']),
                                  pd.DataFrame(np.transpose(linear_model.coef_), columns = ['Coeff'])], axis = 1)
      
        #Accuracy for above model
        from sklearn import metrics
        score = linear_model.score(X_test, y_test)
        
        num_predictions = 100
        diff = 0
        for i in range(num_predictions):
            val = predict[i]
            diff += abs(val - y_test.iloc[i])
            
        #
        textField.Hide()
        textField_1.Hide()       
        textField_2.Show()       
    
        textField_2.AppendText('\n******************************************')
        
        textField_2.AppendText('\n\n     -Model Summary ------ \n')
        textField_2.AppendText(summary)
        
        textField_2.AppendText('\nLinear Regression Accuracy Results >>')
        textField_2.AppendText('\n\n   Features: ['+textField.GetValue()+']')
        textField_2.AppendText('\n    Labels: ['+textField_1.GetValue()+']')

        
        textField_2.AppendText('\n     -Accuracy Score: %f' % (score))
        textField_2.AppendText('\n     -Average prediction difference: %f' % (diff/ num_predictions))
        textField_2.AppendText('\n     -Mean Squared Error: %f' % (metrics.mean_absolute_error(y_test, predict))) 
        textField_2.AppendText('\n     -Mean Absolute Error: %f' % (metrics.mean_squared_error(y_test, predict)))
        textField_2.AppendText('\n     -Root Mean Squared Error:: %f' % (metrics.mean_squared_error(y_test, predict)))
        
        try:
            print(filePath_input)
            print(filePath_input_loc)
            print(filePath_input != "")
        
            if(filePath_input != ""):
                pd.options.mode.chained_assignment = None
                data_en = data_input
        
                for column in data_en.columns:
                    if data_en[column].dtypes == type(object):
                        column_code = column + '_code'
                        data_en[column_code] = linear_le.fit_transform(data_en[column].astype(str))
              
                code_columns = [col for col in data_en.columns if 'code' in col]
                columns = [col for col in data_en.columns if 'code' not in col]
        
                data_pass = data_en[code_columns]
                 #global data_output
                data_output = data_en[columns]
        
                predict_result = linear_model.predict(data_pass)

                data_output['Price'] = predict_result
        
                data_output.to_excel(filePath_input_loc)
                 
        except:
                pass
        
        
      except:
        show_error()
                    
###############################################################################
class DecisionTree():
    def __init__(self):
        pass
    
    #Defining variables
    #criterion  
    criterion = 'gini'
    criterion_choices = ['gini', 'entropy']  
    #random_state
    random_state = 0
    random_state_choices = ['0', '5', '10', '20', '50', '100', '200']    
    #maxdepth
    maxdepth = None
    maxdepth_choices = ['5', '10', '20', '50', '100', '200', '500']    
    #min_samples_leaf        
    sample_leaf = 2
    min_samples_leaf_choices = ['1', '2', '5', '10', '20', '50', '100']
   
    def layout_DecisionTree(self):
        #global Select_criterion
        
        self.Select_criterion = wx.ComboBox(panel, wx.ID_ANY, 'criterion', 
                              choices = self.criterion_choices, 
                              pos = (10, 250), size = (100, 60))
        
        #global Select_random_state
        self.Select_random_state = wx.ComboBox(panel, wx.ID_ANY, 'random_state', 
                              choices = self.random_state_choices, 
                              pos = (130, 250), size = (100, 60))
    
        #global Select_maxdepth 
        self.Select_maxdepth = wx.ComboBox(panel, wx.ID_ANY, 'max_depth', 
                              choices = self.maxdepth_choices, 
                              pos = (250, 250), size = (100, 60))
    
        #global Select_min_samples_leaf
        self.Select_min_samples_leaf = wx.ComboBox(panel, wx.ID_ANY, 'samples_leaf', 
                              choices = self.min_samples_leaf_choices, 
                              pos = (370, 250), size = (100, 60))
    
    
        self.Select_criterion.Bind(wx.EVT_COMBOBOX, self.OnSelectDecisioncriterion)
        self.Select_random_state.Bind(wx.EVT_COMBOBOX, self.OnSelectDecision_randomstate)
        self.Select_maxdepth.Bind(wx.EVT_COMBOBOX, self.OnSelectDecision_maxdepth)
        self.Select_min_samples_leaf.Bind(wx.EVT_COMBOBOX, self.OnSelectDecision_samples_leaf)
        
        self.Select_criterion.Hide()
        self.Select_maxdepth.Hide()
        self.Select_min_samples_leaf.Hide()
        self.Select_random_state.Hide()
    
        
    #Decision Tree Events        
    def OnSelectDecisioncriterion(self, event):
        self.criterion = self.Select_criterion.GetValue()
        self.Decision_Tree()
        
    def OnSelectDecision_randomstate(self, event):
        self.random_state_d = int(self.Select_random_state.GetValue())
        self.Decision_Tree()
        
    def OnSelectDecision_maxdepth(self, event):
        self.maxdepth_d = int(self.Select_maxdepth.GetValue())
        self.Decision_Tree()
        
    def OnSelectDecision_samples_leaf(self, event):
        self.sample_leaf = int(self.Select_min_samples_leaf.GetValue())
        self.Decision_Tree()
        
    
    def Decision_Tree(self):
        #Decision Tree Algo
      try:  
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing
        from matplotlib import pyplot as plt
        from sklearn.tree import DecisionTreeClassifier
        
        
        data_encode = data

        for column in data_encode.columns:
            if data_encode[column].dtype == type(object):
                le = preprocessing.LabelEncoder()
                data_encode[column] = le.fit_transform(data_encode[column].astype(str))
    
        df = data[feature_list]
        y = data[label_list]
        
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test) 
        
        global decision_model
        decision_model = DecisionTreeClassifier(criterion = self.criterion, 
                                     random_state = self.random_state, 
                                     max_depth=self.maxdepth, 
                                     min_samples_leaf=self.sample_leaf)
        
        decision_model.fit(X_train, y_train)
        predict = decision_model.predict(X_test)
        
        #Accuracy for above model
        from sklearn import metrics
        score = decision_model.score(X_test, y_test)
        
        num_predictions = 100
        diff = 0
        for i in range(num_predictions):
            val = predict[i]
            diff += abs(val - y_test.iloc[i])
            
        #
        textField.Hide()
        textField_1.Hide()
        
        textField_2.Show()
    
        textField_2.AppendText('\n******************************************')
        textField_2.AppendText('\nDescision Tree Accuracy Results >>')
        textField_2.AppendText('\n\n   Features: ['+textField.GetValue()+']')
        textField_2.AppendText('\n    Labels: ['+textField_1.GetValue()+']')
        
        textField_2.AppendText('\n    -Accuracy Score: %f' % (score))
        textField_2.AppendText('\n    -Average prediction difference: %f' % (diff/ num_predictions))
        textField_2.AppendText('\n    -Mean Squared Error: %f' % (metrics.mean_absolute_error(y_test, predict))) 
        textField_2.AppendText('\n    -Mean Absolute Error: %f' % (metrics.mean_squared_error(y_test, predict)))
        textField_2.AppendText('\n    -Root Mean Squared Error:: %f' % (metrics.mean_squared_error(y_test, predict)))               
        
        try:
            print(filePath_input)
            print(filePath_input != "")
        
            if(filePath_input != ""):
                pd.options.mode.chained_assignment = None
                data_en = data_input
        
                for column in data_en.columns:
                    if data_en[column].dtypes == type(object):
                        column_code = column + '_code'
                        data_en[column_code] = linear_le.fit_transform(data_en[column].astype(str))
              
                code_columns = [col for col in data_en.columns if 'code' in col]
                columns = [col for col in data_en.columns if 'code' not in col]
        
                data_pass = data_en[code_columns]
                 #global data_output
                data_output = data_en[columns]
        
                predict_result = decision_model.predict(data_pass)

                data_output['Price'] = predict_result
        
                data_output.to_excel(filePath_input_loc)
                 
        except:
                pass
        
      except:
          show_error()
          
    
 
###############################################################################           
class RandomForest():
    def __init__(self):
        pass
    
    #Defining variables
    #criterion
    n_estimators = 100
    n_estimators_choices = ['5', '10', '20', '50', '100', '200', '500']   
    #random_state
    random_state = None
    random_state_choices = ['0', '5', '10', '20', '50', '100', '200']  
    #maxdepth
    maxdepth = None
    maxdepth_choices = ['5', '10', '20', '50', '100', '200', '500']  
    #min_samples_leaf        
    max_features = 'auto'
    max_features_choices = ['auto', 'sqrt', 'log2']

    def layout_RandomForest(self):
        #Adding extra criteria details
        #global Select_n_estimators
  
        self.Select_n_estimators = wx.ComboBox(panel, wx.ID_ANY, 'n_estimators', 
                              choices = self.n_estimators_choices, 
                              pos = (10, 250), size = (100, 60))
        
        #global Select_random_state_decision
        self.Select_random_state_decision = wx.ComboBox(panel, wx.ID_ANY, 'random_state', 
                              choices = self.random_state_choices, 
                              pos = (130, 250), size = (100, 60))
        
        #global Select_maxdepth_decision
        self.Select_maxdepth_decision = wx.ComboBox(panel, wx.ID_ANY, 'max_depth', 
                              choices = self.maxdepth_choices, 
                              pos = (250, 250), size = (100, 60))
        
        #global Select_max_features_decision
        self.Select_max_features_decision = wx.ComboBox(panel, wx.ID_ANY, 'max_features', 
                              choices = self.max_features_choices, 
                              pos = (370, 250), size = (100, 60))
        
        self.Select_n_estimators.Bind(wx.EVT_COMBOBOX, self.OnSelectRandomestimators)
        self.Select_random_state_decision.Bind(wx.EVT_COMBOBOX, self.OnSelectRandom_randomstate)
        self.Select_maxdepth_decision.Bind(wx.EVT_COMBOBOX, self.OnSelectRandom_maxdepth)
        self.Select_max_features_decision.Bind(wx.EVT_COMBOBOX, self.OnSelectRandom_max_features)
        
        self.Select_n_estimators.Hide()
        self.Select_random_state_decision.Hide()
        self.Select_maxdepth_decision.Hide()
        self.Select_max_features_decision.Hide()
        
    #Random Forest Events
    def OnSelectRandomestimators(self, event):
        self.n_estimators = int(self.Select_n_estimators.GetValue())
        self.Random_Forest()
    
    def OnSelectRandom_randomstate(self, event):
        self.random_state = int(self.Select_random_state_decision.GetValue())
        self.Random_Forest()
    
    def OnSelectRandom_maxdepth(self, event):
        self.maxdepth = int(self.Select_maxdepth_decision.GetValue())
        self.Random_Forest()
    
    def OnSelectRandom_max_features(self, event):
        self.max_features = self.Select_max_features_decision.GetValue()
        self.Random_Forest() 
        
    def Random_Forest(self):
        #Implementing RandomForest
      try:  
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing
        from matplotlib import pyplot as plt
        
        data_encode = data

        for column in data_encode.columns:
            if data_encode[column].dtype == type(object):
                le = preprocessing.LabelEncoder()
                data_encode[column] = le.fit_transform(data_encode[column].astype(str))
                
        df = data[feature_list]
        y = data[label_list]
    
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test) 
        
        #Applying RandomForest Algorithm
        from sklearn.ensemble import RandomForestRegressor
        global random_model
        random_model = RandomForestRegressor(n_estimators=self.n_estimators, 
                                      random_state=self.random_state, 
                                      max_depth = self.maxdepth, 
                                      max_features = self.max_features) 
        random_model.fit(X_train, y_train)  
        predict = random_model.predict(X_test)
        
        #Accuracy Score
        from sklearn import metrics
        score = random_model.score(X_test, y_test)
    
        num_predictions = 100
        diff = 0

        for i in range(num_predictions):
            val = predict[i]
            diff += abs(val - y_test.iloc[i])
        
        #Text field
        textField.Hide()
        textField_1.Hide()     
        textField_2.Show()

        textField_2.AppendText('\n******************************************')
        textField_2.AppendText('\nRandom Forest Accuracy Results >>')
        textField_2.AppendText('\n\n   Features: ['+textField.GetValue()+']')
        textField_2.AppendText('\n    Labels: ['+textField_1.GetValue()+']')
        
        textField_2.AppendText('\n     -Accuracy Score: %f' % (score))
        textField_2.AppendText('\n     -Average prediction difference: %f' % (diff/ num_predictions))
        textField_2.AppendText('\n     -Mean Squared Error: %f' % (metrics.mean_absolute_error(y_test, predict))) 
        textField_2.AppendText('\n     -Mean Absolute Error: %f' % (metrics.mean_squared_error(y_test, predict)))
        textField_2.AppendText('\n     -Root Mean Squared Error:: %f' % (metrics.mean_squared_error(y_test, predict)))
        
        try:
            print(filePath_input)
            print(filePath_input != "")
        
            if(filePath_input != ""):
                pd.options.mode.chained_assignment = None
                data_en = data_input
        
                for column in data_en.columns:
                    if data_en[column].dtypes == type(object):
                        column_code = column + '_code'
                        data_en[column_code] = linear_le.fit_transform(data_en[column].astype(str))
              
                code_columns = [col for col in data_en.columns if 'code' in col]
                columns = [col for col in data_en.columns if 'code' not in col]
        
                data_pass = data_en[code_columns]
                 #global data_output
                data_output = data_en[columns]
        
                predict_result = random_model.predict(data_pass)

                data_output['Price'] = predict_result
        
                data_output.to_excel(filePath_input_loc)
                 
        except:
             pass
        
     
      except:
        show_error()

class Logistic_Regression():
    def __init__():
        pass
    
    def Logistic__Regression():
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        
        data_encode = data

        for column in data_encode.columns:
            if data_encode[column].dtype == type(object):
                data_encode[column] = le.fit_transform(data_encode[column].astype(str))
    
        df = data[feature_list]
        y = data[label_list]
        
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test) 
        
        #Here Trying to implement model
        import statsmodels.api as sm
        logit_model = sm.Logit(y_train, X_train)
        result = logit_model.fit()
        #print(result.summary2())
        summary = str(result.summary2())
        
        #Logistic Regression
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        predict = logreg.predict(X_test)
        
        #accuracy score
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, predict)
        
        textField.Hide()
        textField_1.Hide()       
        textField_2.Show()       
    
        textField_2.AppendText('\n******************************************')

        textField_2.AppendText('\n\n     -Model Summary ------ \n')
        textField_2.AppendText(summary)
        
        textField_2.AppendText('\nLogistic Regression Accuracy Results >>')
        textField_2.AppendText('\n\n   Features: ['+textField.GetValue()+']')
        textField_2.AppendText('\n    Labels: ['+textField_1.GetValue()+']')
      
        textField_2.AppendText('\n     -Accuracy Score: %f' % (score))
        
         
    
class KNearestNeighbour():
    def __init__(self):
        pass
    
    def KNearest_Neighbour():
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split
        
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        
        data_encode = data

        for column in data_encode.columns:
            if data_encode[column].dtype == type(object):
                data_encode[column] = le.fit_transform(data_encode[column].astype(str))
    
        df = data[feature_list]
        y = data[label_list]
        
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test) 
        
        #Modeling
        k = 5
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        predict = classifier.predict(X_test)
        
        #accuracy score
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, predict)
        
        textField.Hide()
        textField_1.Hide()       
        textField_2.Show()       
    
        textField_2.AppendText('\n******************************************')
        
        textField_2.AppendText('\nKNN Accuracy Results >>')
        textField_2.AppendText('\n\n   Features: ['+textField.GetValue()+']')
        textField_2.AppendText('\n    Labels: ['+textField_1.GetValue()+']')
        textField_2.AppendText('\n    K: '+str(k))
      
        textField_2.AppendText('\n     -Accuracy Score: %f' % (score))
        
class Naivebayes():
    def __init__(self):
        pass
    
    def Naive_bayes():
        from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
        from sklearn.model_selection import train_test_split
        
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        
        data_encode = data

        for column in data_encode.columns:
            if data_encode[column].dtype == type(object):
                data_encode[column] = le.fit_transform(data_encode[column].astype(str))
    
        df = data[feature_list]
        y = data[label_list]
        
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test) 
        
        #Modeling
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        predict_g = gnb.predict(X_test)
        
        #accuracy score
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, predict_g)
        
        textField.Hide()
        textField_1.Hide()       
        textField_2.Show()       
    
        textField_2.AppendText('\n******************************************')
        
        textField_2.AppendText('\nNaive bayes Accuracy Results >>')
        textField_2.AppendText('\n\n   Features: ['+textField.GetValue()+']')
        textField_2.AppendText('\n    Labels: ['+textField_1.GetValue()+']')
      
        textField_2.AppendText('\n     -Accuracy Score: %f' % (score))
        
class Xgboost():
    def __init__(self):
        pass
    
    def XGBoost():
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        
        data_encode = data

        for column in data_encode.columns:
            if data_encode[column].dtype == type(object):
                data_encode[column] = le.fit_transform(data_encode[column].astype(str))
    
        df = data[feature_list]
        y = data[label_list]
        
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test) 
        
        #Modeling
        xgb = XGBClassifier(min_child_weight=2, max_depth = 10, 
                    max_delta_step = 1, colsample_bytree = 1)
        xgb.fit(X_train, y_train)
        predict_x = xgb.predict(X_test)

        #accuracy score
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, predict_x)
        
        textField.Hide()
        textField_1.Hide()       
        textField_2.Show()       
    
        textField_2.AppendText('\n******************************************')
        
        textField_2.AppendText('\nXGBoost Accuracy Results >>')
        textField_2.AppendText('\n\n   Features: ['+textField.GetValue()+']')
        textField_2.AppendText('\n    Labels: ['+textField_1.GetValue()+']')
      
        textField_2.AppendText('\n     -Accuracy Score: %f' % (score))
        
class RandomForestClassifier():
    def __init__(self):
        pass
    def RandomForest():
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        
        data_encode = data

        for column in data_encode.columns:
            if data_encode[column].dtype == type(object):
                data_encode[column] = le.fit_transform(data_encode[column].astype(str))
    
        df = data[feature_list]
        y = data[label_list]
        
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test)
        
        #Modeling        
        clf = RandomForestClassifier(n_estimators=100, random_state=20, max_depth = 20, max_features = 2)
        clf.fit(X_train, y_train)

        predict_c = clf.predict(X_test)

        #accuracy score
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, predict_c)
        
        textField.Hide()
        textField_1.Hide()       
        textField_2.Show()       
    
        textField_2.AppendText('\n******************************************')
        
        textField_2.AppendText('\nRandomForest Classifier Accuracy Results >>')
        textField_2.AppendText('\n\n   Features: ['+textField.GetValue()+']')
        textField_2.AppendText('\n    Labels: ['+textField_1.GetValue()+']')
      
        textField_2.AppendText('\n     -Accuracy Score: %f' % (score))
        
class LightGBM():
    def __init__(self):
        pass
    
    def Light_GBM():
        import lightgbm 
        from sklearn.model_selection import train_test_split
        
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        
        data_encode = data

        for column in data_encode.columns:
            if data_encode[column].dtype == type(object):
                data_encode[column] = le.fit_transform(data_encode[column].astype(str))
    
        df = data[feature_list]
        y = data[label_list]
        
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test)
        
        #Random
        d_train = lightgbm.Dataset(X_train, label = y_train)

        params = {}
        params['learning_rate'] = 0.003
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'binary'
        params['metric'] = 'binary_logloss'
        params['sub_feature'] = 0.5
        params['num_leaves'] = 10
        params['min_data'] = 50
        params['max_depth'] = 10

        lgb = lightgbm.train(params, d_train, 100)

        predict_l = lgb.predict(X_test)

        #convert into binary values
        for i in range(len(predict_l)):
            if predict_l[i]>=.5:       # setting threshold to .5
                predict_l[i]=1
            else:  
                predict_l[i]=0

        #accuracy score
        from sklearn.metrics import accuracy_score
        score = accuracy_score(predict_l, y_test)
        
        textField.Hide()
        textField_1.Hide()       
        textField_2.Show()       
    
        textField_2.AppendText('\n******************************************')
        
        textField_2.AppendText('\nLightGBM Accuracy Results >>')
        textField_2.AppendText('\n\n   Features: ['+textField.GetValue()+']')
        textField_2.AppendText('\n    Labels: ['+textField_1.GetValue()+']')
      
        textField_2.AppendText('\n     -Accuracy Score: %f' % (score))
        
    
##############################################################################
#Lets begin the fun part : Neural Network
class Keras_NeuralNetwork():
    def __init__(self):
        pass
    
    #Defining variable
    #For now, Let's keep Sequential as default model
    model = 'Sequential'
    model_choices = ['Sequential', 'Model']
    
    layers = 'Core - Dense'
    layer_choices = ['Dense', 'Conv1D', 'MaxPooling1D', 'LocallyConnected1D', 'RNN', 'Embedding', 'LeakyReLU', 'BatchNormalization']
    
    noise = 'GaussianNoise'
    noise_choices = ['GaussianNoise', 'GaussianDropOut', 'AlphaDropOut']
    
    layer_wrapper = 'TimeDistributed'
    layer_wrapper_choices = ['TimeDistributed', 'Bidirectional']
    
    activation = 'softmax'
    activation_choices = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'hard_sigmoid', 'exponential', 'linear']
    
    optimizer = 'sgd'
    optimizer_choices = ['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    
    loss = 'mean_squared_error'
    loss_choices = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']
    
    metrics = 'binary_accuracy'
    metrics_choices = ['binary_accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy', 'top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
    
    def layout_Keras_NeuralNetwork_layer(self):

        #layers
        self.Select_layer = wx.ComboBox(panel, wx.ID_ANY, 'Layer', 
                              choices = self.layer_choices, 
                              pos = (10, 250), size = (100, 60))
        
        #activation and Input size - other details
        self.Select_activation = wx.ComboBox(panel, wx.ID_ANY, 'Activation', 
                              choices = self.activation_choices, 
                              pos = (370, 250), size = (100, 60))
                        
        
        self.Select_layer.Bind(wx.EVT_COMBOBOX, self.OnSelect_layer)
        self.Select_activation.Bind(wx.EVT_COMBOBOX, self.OnSelect_activation)
        
        self.Select_layer.Hide()
        self.Select_activation.Hide()
        
    def OnSelect_layer(self, event):
        self.layers = self.Select_layer.GetValue()
    
    def OnSelect_activation(self, event):
        self.activation = self.Select_activation.GetValue()
        
    def layout_Keras_NeuralNetwork_compile(self):
        
        
        #optimizer
        self.Select_optimizer = wx.ComboBox(panel, wx.ID_ANY, 'optimizer', 
                              choices = self.optimizer_choices, 
                              pos = (10, 250), size = (140, 60))
        
        #loss
        self.Select_loss = wx.ComboBox(panel, wx.ID_ANY, 'loss', 
                              choices = self.loss_choices, 
                              pos = (180, 250), size = (140, 60))
        
        #metrics
        self.Select_metrics = wx.ComboBox(panel, wx.ID_ANY, 'metrics', 
                              choices = self.metrics_choices, 
                              pos = (350, 250), size = (140, 60))
        
        self.Select_optimizer.Bind(wx.EVT_COMBOBOX, self.OnSelect_optimizer)
        self.Select_loss.Bind(wx.EVT_COMBOBOX, self.OnSelect_loss)
        self.Select_metrics.Bind(wx.EVT_COMBOBOX, self.OnSelect_metrics)
        
        self.Select_optimizer.Hide()
        self.Select_loss.Hide()
        self.Select_metrics.Hide()
        
    def OnSelect_optimizer(self, event):
        self.optimizer = self.Select_optimizer.GetValue()
        self.Neural_Network()
    
    def OnSelect_loss(self, event):
        self.loss = self.Select_loss.GetValue()
        self.Neural_Network()
    
    def OnSelect_metrics(self, event):
        self.metrics = self.Select_metrics.GetValue()
        self.Neural_Network()
    
    def Neural_Network(self):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn import preprocessing
            from matplotlib import pyplot as plt
        
            data_encode = data

            for column in data_encode.columns:
                if data_encode[column].dtype == type(object):
                    le = preprocessing.LabelEncoder()
                    data_encode[column] = le.fit_transform(data_encode[column].astype(str))
                
            df = data[feature_list]
            y = data[label_list]
    
            X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        
            #Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()  
            X_train = sc.fit_transform(X_train)  
            X_test = sc.transform(X_test) 
        
            #Applying Keras Neural Network things
            from keras.models import Sequential
            model = Sequential()
            
            #Add layers -- This on is tricky, as all layers needs different setting
            from keras import layers
            
            for index in range(len(lst_layer)):
                if 'Dense' in lst_layer:
                    model.add(layers.Dense(units=lst_size[index], activation=lst_activation[index], input_dim=lst_Input_dim[index]))
                    
            model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)
            
            model.fit(X_train, y_train, epochs=2, batch_size=10)
            
            #Accuracy Score
            score = model.evaluate(X_test, y_test, batch_size = 10)
            
            #Text field
            textField.Hide()
            textField_1.Hide()     
            textField_2.Show()

            textField_2.AppendText('\n******************************************')
            textField_2.AppendText('\nRandom Forest Accuracy Results >>')
            textField_2.AppendText('\n\n   Features: ['+textField.GetValue()+']')
            textField_2.AppendText('\n    Labels: ['+textField_1.GetValue()+']')
        
            textField_2.AppendText('\n     -Accuracy Score: %f' % (score))
                
            
            

            
        except:
            pass
            
        
    
    def Neural_Network_Classifier(self):
        pass
        
       
class Keras_NeuralNetwork_classifier():
    def __init__(self):
        pass
    
    #Defining variable
    model = 'Sequential'
    model_choices = ['Sequential', 'Model']
    
    layers = 'Core - Dense'
    layer_choices = ['Dense', 'Conv1D', 'MaxPooling1D', 'LocallyConnected1D', 'RNN', 'Embedding', 'LeakyReLU', 'BatchNormalization']
    
    noise = 'GaussianNoise'
    noise_choices = ['GaussianNoise', 'GaussianDropOut', 'AlphaDropOut']
    
    layer_wrapper = 'TimeDistributed'
    layer_wrapper_choices = ['TimeDistributed', 'Bidirectional']
    
    activation = 'softmax'
    activation_choices = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'hard_sigmoid', 'exponential', 'linear']
    
    optimizer = 'sgd'
    optimizer_choices = ['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    
    loss = 'mean_squared_error'
    loss_choices = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']
    
    metrics = 'binary_accuracy'
    metrics_choices = ['binary_accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy', 'top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
    
    
    
    


















        
        
        




        
#from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
##from matplotlib.backends.backend_wx import NavigationToolbar2Wx
#from matplotlib.figure import Figure
#import numpy as np
#
#class CanvasPanel(wx.Panel):
#    def __init__(self, parent):
#        wx.Panel.__init__(self, parent)
#        self.figure = Figure()
#        self.axes = self.figure.add_subplot(111)
#        self.canvas = FigureCanvas(self, -1, self.figure)
#        self.sizer = wx.BoxSizer(wx.VERTICAL)
#        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
#        self.SetSizer(self.sizer)
#        self.Fit()
#
#    def draw(self, plt):
#        t = np.arange(0.0, 3.0, 0.01)
#        s = np.sin(2 * np.pi * t)
#        self.axes.plot(t, s)

app = wx.App()

frame = wx.Frame(None, -1, 'Next.py')
frame.SetDimensions(0, 0, 1100, 650)
panel = wx.Panel(frame, wx.ID_ANY)

wx.StaticText(
    panel, -1, "Select File, Enter Features and Labels and then click :", pos=(10, 10))

#Trying to break panel in two parts
wx.StaticText(panel, -1, "|", pos=(540, 580))
wx.StaticText(panel, -1, "|", pos=(540, 560))
wx.StaticText(panel, -1, "|", pos=(540, 542))
wx.StaticText(panel, -1, "|", pos=(540, 520))
wx.StaticText(panel, -1, "|", pos=(540, 500))
wx.StaticText(panel, -1, "|", pos=(540, 480))
wx.StaticText(panel, -1, "|", pos=(540, 460))
wx.StaticText(panel, -1, "|", pos=(540, 440))
wx.StaticText(panel, -1, "|", pos=(540, 420))
wx.StaticText(panel, -1, "|", pos=(540, 400))
wx.StaticText(panel, -1, "|", pos=(540, 380))
wx.StaticText(panel, -1, "|", pos=(540, 360))
wx.StaticText(panel, -1, "|", pos=(540, 340))
wx.StaticText(panel, -1, "|", pos=(540, 320))
wx.StaticText(panel, -1, "|", pos=(540, 300))
wx.StaticText(panel, -1, "|", pos=(540, 280))
wx.StaticText(panel, -1, "|", pos=(540, 260))
wx.StaticText(panel, -1, "|", pos=(540, 240))
wx.StaticText(panel, -1, "|", pos=(540, 220))
wx.StaticText(panel, -1, "|", pos=(540, 200))
wx.StaticText(panel, -1, "|", pos=(540, 180))
wx.StaticText(panel, -1, "|", pos=(540, 160))
wx.StaticText(panel, -1, "|", pos=(540, 140))
wx.StaticText(panel, -1, "|", pos=(540, 120))
wx.StaticText(panel, -1, "|", pos=(540, 100))
wx.StaticText(panel, -1, "|", pos=(540, 80))
wx.StaticText(panel, -1, "|", pos=(540, 60))
wx.StaticText(panel, -1, "|", pos=(540, 40))
wx.StaticText(panel, -1, "|", pos=(540, 20))



#Upload Button
upload_button = wx.Button(panel, wx.ID_ANY, 'Select report file >>', (10, 90))

upload_button.Bind(wx.EVT_BUTTON, onUploadButton)

#coeff button
coeff_button = wx.Button(panel, wx.ID_ANY, 'coeff >>', (190, 210))

coeff_button.Bind(wx.EVT_BUTTON, onCoeffButton)

coeff_button.Hide()



#Upload Input button
upload_Input_button = wx.Button(panel, wx.ID_ANY, 'Upload Inputs >>', (10, 285))

upload_Input_button.Bind(wx.EVT_BUTTON, On_upload_Input_button)
upload_Input_button.Hide()
    
#Download Results button
download_results_button = wx.Button(panel, wx.ID_ANY, 'Download Results >>', (140, 285))

download_results_button.Bind(wx.EVT_BUTTON, On_download_results_button)
download_results_button.Hide()

#Download Notes    
download_notes_button = wx.Button(panel, wx.ID_ANY, 'Download Notes >>', (10, 510))

download_notes_button.Bind(wx.EVT_BUTTON, On_download_notes_button)

#Add button - Neural Neywork -->  Add Layers
Add_button = wx.Button(panel, wx.ID_ANY, 'Add Layer >>', (10, 285))
Add_button.Hide()

#Lock button - Neural Network --> Lock Layer
Lock_button = wx.Button(panel, wx.ID_ANY, 'Lock Layers >>', (140, 285))
Lock_button.Hide()

#Refersh
def On_refersh_button(event):
    download_results_button.show()
    
refersh_button = wx.Button(panel, wx.ID_ANY, 'Refersh', (423, 510))

refersh_button.Bind(wx.EVT_BUTTON, On_download_notes_button)


#text fields
textField = wx.TextCtrl(panel, -1, "No File Selected yet!!!",
                        pos=(10, 320), size=(500, 80))

textField_1 = wx.TextCtrl(panel, -1, "No File Selected yet!!!",
                        pos=(10, 420), size=(500, 80))

textField_2 = wx.TextCtrl(panel, -1,
                        pos=(10, 320), size=(520, 180), style= wx.TE_MULTILINE)

textField_2.Hide()

#Neural - Text Fields
#Select Size and Input_dim of your select Layer
textField_size = wx.TextCtrl(panel, -1, "Size",pos=(140, 250), size=(80, 24))
textField_Input_dim = wx.TextCtrl(panel, -1, "Input_dim",pos=(260, 250), size=(80, 24))
textField_total_layers = wx.TextCtrl(panel, -1, " - -",pos=(450, 210), size=(25,24))

textField_size.Hide()
textField_Input_dim.Hide()
textField_total_layers.Hide()

frame.Show()
frame.Centre()
app.MainLoop()

del app

