from multimodel_fusion import multimodel_fusion
from save_load import load, save
from Data_gen import *
from plot_result import *
from Classification import *

def full_analysis():
    datagen()
    X_train_Load_Forecasting=load('X_train_Load_Forecasting')
    X_test_Load_Forecasting=load('X_test_Load_Forecasting')
    y_train_Load_Forecasting=load('y_train_Load_Forecasting')
    y_test_Load_Forecasting=load('y_test_Load_Forecasting')

    X_train_Fault_Detection=load('X_train_Fault_Detection')
    X_test_Fault_Detection=load('X_test_Fault_Detection')
    y_train_Fault_Detection=load('y_train_Fault_Detection')
    y_test_Fault_Detection=load('y_test_Fault_Detection')

    #Fault Detection

    #PROPOSED
    reg_met, cls_met= multimodel_fusion()
    save('proposed_cls_met',cls_met)

    #CNN-LSTM
    met = cnn_lstm_cls(X_train_Fault_Detection, X_test_Fault_Detection,  y_train_Fault_Detection,y_test_Fault_Detection)
    save('CNN-LSTM_cls_met', met)

    #LSTM-ANN
    cm =lstm_ann_cls(X_train_Fault_Detection, X_test_Fault_Detection,  y_train_Fault_Detection,y_test_Fault_Detection)
    save('LSTM-ANN_cls_met',cm)

    #MKCNN
    cm =mkcnn_cls(X_train_Fault_Detection, X_test_Fault_Detection,  y_train_Fault_Detection,y_test_Fault_Detection)
    save('MKCNN_cls_met', cm)

    #GDNN
    cm =dnn_model_cls(X_train_Fault_Detection, X_test_Fault_Detection,  y_train_Fault_Detection,y_test_Fault_Detection)
    save('GDNN_cls_met', cm)

    #Load Forecasting

    #PROPOSED
    reg_met, cls_met = multimodel_fusion()
    save('proposed_reg_met', reg_met)

    #CNN-LSTM
    met = cnn_lstm_reg(X_train_Load_Forecasting, X_test_Load_Forecasting,  y_train_Load_Forecasting,y_test_Load_Forecasting)
    save('CNN-LSTM_reg_met', met)

    #LSTM-ANN
    cm =lstm_ann_reg(X_train_Load_Forecasting, X_test_Load_Forecasting,  y_train_Load_Forecasting,y_test_Load_Forecasting)
    save('LSTM-ANN_reg_met',cm)

    #MKCNN
    cm =mkcnn_reg(X_train_Load_Forecasting, X_test_Load_Forecasting,  y_train_Load_Forecasting,y_test_Load_Forecasting)
    save('MKCNN_reg_met', cm)

    #GDNN
    cm =dnn_model_reg(X_train_Load_Forecasting, X_test_Load_Forecasting,  y_train_Load_Forecasting,y_test_Load_Forecasting)
    save('GDNN_reg_met', cm)


a =0
if a == 1:
    full_analysis()
polt_res_Load_Forecastin()
polt_res_Fault_Detection()