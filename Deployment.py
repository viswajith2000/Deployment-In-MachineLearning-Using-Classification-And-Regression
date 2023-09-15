from tkinter import *
from tkinter import ttk

import numpy as np
import pandas as pd
from tkinter import filedialog

# ML agos
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


class base():
    def __init__(self):
        print("Base started")

    def base_fn(self):
        ob = home_page()
        ob.home_fn()


class home_page(base):
    global data, path

    def home_fn(self):
        global x_index, y_index, selected_x, selected_y
        x_index = []
        y_index = []
        selected_x = []
        selected_y = []

        def select_x_fn():
            # Here the selected one from the combobox will get and located the index num in dataset and the index num will be appended in list_index as X var
            x = select_col.get()
            # print("selected column = ",x)
            selected_x.append(x)

            ind_x = data.columns.get_loc(x)
            # print("X cols-------->",ind_x)
            x_index.append(ind_x)

            # inserting to the text box
            x_txt_box.insert(END, x + '\n')

        def select_y_fn():
            # Here the selected one from the combobox will get and located the index num in dataset and the index num will be appended in list_index as Y var
            y = select_col.get()
            selected_y.append(y)
            # print("selected column = ",y)
            ind_y = data.columns.get_loc(y)
            # print("Y cols-------->",ind_y)
            y_index.append(ind_y)
            y_txt_box.insert(END, y + '\n')

        def page_2_fn():
            # this fn to
            # print(list_index)
            x_col = x_index
            y_col = y_index
            name_x = selected_x
            name_y = selected_y
            win.destroy()
            obj = algorithm_choose()
            obj.algorithms(path, x_col, y_col, name_x, name_y)

        def open_file_fn():
            global path, data
            file = filedialog.askopenfilename(title="Open File", filetypes=(
            ("csv files", "*.csv"), ("Text Files", "*.txt"), ("All files", "*."), ("Python files", "*.py")))
            #print(file)
            
            path = file.replace("/", "//")
            # print(path)
            data = pd.read_csv(path)
            list_1 = list(data.columns)
            #print(list_1)
            select_col['values'] = tuple(list_1)
            

        def clear_fn():
            x_txt_box.delete('1.0', END)
            y_txt_box.delete('1.0', END)
            y_index.clear()
            x_index.clear()
            selected_x.clear()
            selected_y.clear()

        win = Tk()
        win.geometry('1400x1400')
        win.title("Prediction Algorithms")

        frame1 = Frame(win, bg='black')
        frame1.pack(ipadx=50, ipady=50, expand=True, fill='both')

        header_l = Label(frame1, text='PREDICTIONS', font=("times new roman", "28", "bold"), bg='red', relief=RAISED, padx=6)
        header_l.pack(ipady=10)

        select_col_en = StringVar()
        # keepvalue=month_en.get()
        select_col = ttk.Combobox(frame1, width=30, textvariable=select_col_en)
        select_col.place(x=570, y=180)

        # -->to append the values in combobox

        select_x_btn = Button(frame1, text='independent', relief=GROOVE, command=select_x_fn)
        select_x_btn.place(x=300, y=300)

        select_y_btn = Button(frame1, text='dependent', relief=GROOVE, command=select_y_fn)
        select_y_btn.place(x=950,y=300)

        x_txt_box = Text(frame1, font=("times new roman", '14', "bold"), width=30, height=10)
        x_txt_box.place(x=150, y=350)

        y_txt_box = Text(frame1, font=("times new roman", '14', "bold"), width=30, height=10)
        y_txt_box.place(x=850, y=350)

        next_btn = Button(frame1, text='next', relief=GROOVE, command=page_2_fn)
        next_btn.place(x=650, y=400)

        refresh_btn = Button(frame1, text='clear', relief=GROOVE, command=clear_fn)
        refresh_btn.place(x=650, y=450)

        open_btn = Button(frame1, text='Open', relief=GROOVE, command=open_file_fn)
        open_btn.place(x=650, y=150)

        # month.current(12)

        win.mainloop()


class algorithm_choose(home_page):

    def algorithms(self, path, x_col, y_col, name_x, name_y):
        win1 = Tk()
        win1.geometry('1400x1400')
        # win1.state('zoomed')
        win1.title("Prediction Algorithms")
        win1.config(background='black')
        frame2 = Frame(win1, bg='black')
        frame2.pack(ipadx=50, ipady=50, expand=True, fill='both')

        def slin_reg_fn():  # Simple linear regression

            frame2.destroy()

            header_l = Label(win1, text='Simple-Linear Regression', font=("times new roman", "28", "bold"), bg='red',
                              relief=RAISED, padx=6)
            header_l.pack(ipady=10)

            frame5 = Frame(win1, bg='white')
            frame5.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_slin_reg_fn():  # Simple linear regression model
                entry = float(col_1_e.get())
                # print(path, x_col, y_col,name_x,name_y)

                # Train splitting
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

                # Training model
                regressor = LinearRegression()
                regressor.fit(x_train, y_train)

                y_pred = regressor.predict(x_test)
                pred_y = regressor.predict([[entry]])
                print(pred_y)

                pred_l = Label(frame5, text=pred_y[0][0].round(2), font=("times new roman", "18", "bold"), bg='light grey')
                pred_l.place(x=620, y=250)

            col_1_l = Label(frame5, text=f'{name_x[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_1_l.place(x=370, y=150)

            col_1_en = StringVar()
            col_1_e = Entry(frame5, width=20, textvariable=col_1_en)
            col_1_e.place(x=620, y=155)

            result_l = Label(frame5, text=f"{name_y[0]}", font=("times new roman", "18", "bold"), bg='light grey')
            result_l.place(x=370, y=250)

            result_btn = Button(frame5, text='Predict', bg='light grey', relief=GROOVE, command=pred_slin_reg_fn)
            result_btn.place(x=510, y=355)

        def mlin_reg_fn():  # Multi linear regression
            frame2.destroy()

            header_l = Label(win1, text='Multi-Linear Regression', font=("times new roman", "28", "bold"), bg='red',
                              relief=RAISED, padx=6)
            header_l.pack(ipady=10)

            frame6 = Frame(win1, bg='white')
            frame6.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_mlin_reg_fn():  # Multi linear regression Model

                entry1 = float(col_1_e.get())
                entry2 = float(col_2_e.get())

                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

                regressor = LinearRegression()
                regressor.fit(x_train, y_train)

                y_pred = regressor.predict(x_test)
                # print(y_pred)

                pred_y = regressor.predict([[entry1, entry2]])
                print(pred_y)

                pred_l = Label(frame6, text=pred_y[0][0].round(2), font=("times new roman", "18", "bold"), bg='light grey')
                pred_l.place(x=600, y=300)

            col_1_l = Label(frame6, text=f'{name_x[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_1_l.place(x=350, y=150)

            col_2_l = Label(frame6, text=f'{name_x[1]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_2_l.place(x=350, y=210)

            col_3_l = Label(frame6, text=f'{name_y[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_3_l.place(x=350, y=300)

            col_1_en = StringVar()
            col_1_e = Entry(frame6, width=20, textvariable=col_1_en)
            col_1_e.place(x=600, y=155)

            col_2_en = StringVar()
            col_2_e = Entry(frame6, width=20, textvariable=col_2_en)
            col_2_e.place(x=600, y=210)

            result_btn = Button(frame6, text='Predict', bg='light grey', relief=GROOVE, command=pred_mlin_reg_fn)
            result_btn.place(x=460, y=375)

        def plin_reg_fn():
            frame2.destroy()

            header_l = Label(win1, text='Polynomial Regression', font=("times new roman", "28", "bold"), bg='red', relief=RAISED,
                              padx=6)
            header_l.pack(ipady=10)

            frame7 = Frame(win1, bg='white')
            frame7.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_poly_reg_fn():
                x_entry =float(col_1_e.get())
                poly_reg = PolynomialFeatures(degree=7)
                x1 = poly_reg.fit_transform(x)

                regressor = LinearRegression()
                regressor.fit(x1, y)

                pred_y = regressor.predict(poly_reg.fit_transform([[x_entry]]))
                print(pred_y)

                pred_l = Label(frame7, text=pred_y[0][0].round(2), font=("times new roman", "18", "bold"), bg='light grey')
                pred_l.place(x=600, y=250)

            col_1_l = Label(frame7, text=f'{name_x[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_1_l.place(x=350, y=150)

            col_1_en = StringVar()
            col_1_e = Entry(frame7, width=20, textvariable=col_1_en)
            col_1_e.place(x=600, y=155)

            result_l = Label(frame7, text=f"{name_y[0]}", font=("times new roman", "18", "bold"), bg='light grey')
            result_l.place(x=350, y=250)

            result_btn = Button(frame7, text='Predict', bg='light grey', relief=GROOVE, command=pred_poly_reg_fn)
            result_btn.place(x=460, y=355)

        # Classification------------------

        def lcls_fn():
            frame2.destroy()

            header_l = Label(win1, text='Logistic Regression', font=("times new roman", "28", "bold"), bg='red', relief=RAISED,
                              padx=6)
            header_l.pack(ipady=10)

            frame8 = Frame(win1, bg='white')
            frame8.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_logcls_fn():
                x_entry1 = col_1_e.get()
                x_entry2 = col_2_e.get()

                sc = StandardScaler()
                x1 = sc.fit_transform(x)
                x2 = sc.fit_transform([[x_entry1, x_entry2]])  # fit transform user input

                # x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.30,random_state=0)

                classifier = LogisticRegression()
                classifier.fit(x1, np.ravel(y))

                pred_y = classifier.predict(x2)

                pred_l = Label(frame8, text=pred_y[0], font=("times new roman", "18", "bold"), bg='light grey')
                pred_l.place(x=600, y=300)
                # print(pred_y)

            col_1_l = Label(frame8, text=f'{name_x[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_1_l.place(x=350, y=150)

            col_2_l = Label(frame8, text=f'{name_x[1]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_2_l.place(x=350, y=210)

            col_3_l = Label(frame8, text=f'{name_y[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_3_l.place(x=350, y=300)

            col_1_en = StringVar()
            col_1_e = Entry(frame8, width=20, textvariable=col_1_en)
            col_1_e.place(x=600, y=155)

            col_2_en = StringVar()
            col_2_e = Entry(frame8, width=20, textvariable=col_2_en)
            col_2_e.place(x=600, y=210)

            result_btn = Button(frame8, text='Predict', bg='light grey', relief=GROOVE, command=pred_logcls_fn)
            result_btn.place(x=460, y=375)

        def naive_fn():
            frame2.destroy()

            header_l = Label(win1, text='Naive Bayes', font=("times new roman", "28", "bold"), bg='red', relief=RAISED, padx=6)
            header_l.pack(ipady=10)

            frame9 = Frame(win1, bg='white')
            frame9.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_nvbys_fn():
                x_entry1 = float(col_1_e.get())
                x_entry2 = float(col_2_e.get())

                classifier = GaussianNB()
                classifier.fit(x, y)
                

                pred_y = classifier.predict([[x_entry1, x_entry2]])
                print(pred_y)

                pred_l = Label(frame9, text=pred_y[0], font=("Futura", "18", "bold"), bg='light grey')
                pred_l.place(x=600, y=300)

            col_1_l = Label(frame9, text=f'{name_x[0]}', font=("Futura", "18", "bold"), bg='light grey')
            col_1_l.place(x=350, y=150)

            col_2_l = Label(frame9, text=f'{name_x[1]}', font=("Futura", "18", "bold"), bg='light grey')
            col_2_l.place(x=350, y=210)

            col_3_l = Label(frame9, text=f'{name_y[0]}', font=("Futura", "18", "bold"), bg='light grey')
            col_3_l.place(x=350, y=300)

            col_1_en = StringVar()
            col_1_e = Entry(frame9, width=20, textvariable=col_1_en)
            col_1_e.place(x=600, y=155)

            col_2_en = StringVar()
            col_2_e = Entry(frame9, width=20, textvariable=col_2_en)
            col_2_e.place(x=600, y=210)

            result_btn = Button(frame9, text='Predict', bg='light grey', relief=GROOVE, command=pred_nvbys_fn)
            result_btn.place(x=460, y=375)
            
        def knn_fn():
            frame2.destroy()

            header_l = Label(win1, text='KNN Algorithm', font=("times new roman", "28", "bold"), bg='red', relief=RAISED, padx=6)
            header_l.pack(ipady=10)

            frame10 = Frame(win1, bg='white')
            frame10.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_knn():
                x_entry1 = col_1_e.get()
                x_entry2 = col_2_e.get()
               
                from sklearn.preprocessing import StandardScaler
                sc=StandardScaler()
               

                from sklearn.model_selection import train_test_split
                x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

                from sklearn.neighbors import KNeighborsClassifier
                classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
                classifier.fit(x_train,y_train)
                y_pred=classifier.predict(x_test)
                
                pred_l = Label(frame10, text=y_pred[0], font=("times new roman", "18", "bold"), bg='light grey')
                pred_l.place(x=600, y=300)
                
            col_1_l = Label(frame10, text=f'{name_x[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_1_l.place(x=350, y=150)
    
            col_2_l = Label(frame10, text=f'{name_x[1]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_2_l.place(x=350, y=210)
    
            col_3_l = Label(frame10, text=f'{name_y[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_3_l.place(x=350, y=300)
    
            col_1_en = StringVar()
            col_1_e = Entry(frame10, width=20, textvariable=col_1_en)
            col_1_e.place(x=600, y=155)
    
            col_2_en = StringVar()
            col_2_e = Entry(frame10, width=20, textvariable=col_2_en)
            col_2_e.place(x=600, y=210)

                
            result_btn = Button(frame10, text='Predict', bg='light grey', relief=GROOVE, command=pred_knn)
            result_btn.place(x=460, y=375)
        def svm_fn():
            frame2.destroy()

            header_l = Label(win1, text='SVM', font=("times new roman", "28", "bold"), bg='red', relief=RAISED, padx=6)
            header_l.pack(ipady=10)

            frame11 = Frame(win1, bg='white')
            frame11.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_svm():
                x_entry1 = col_1_e.get()
                x_entry2 = col_2_e.get()
                 
                from sklearn.preprocessing import StandardScaler
                sc=StandardScaler()
                # x=sc.fit_transform(x)

                from sklearn.model_selection import train_test_split
                x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

                from sklearn.svm import SVC
                classifier=SVC(C=1,kernel='rbf',random_state=0,gamma=0.95)
                classifier.fit(x_train,y_train)

                y_pred=classifier.predict(x_test)
                
                
                pred_l = Label(frame11, text=y_pred[0], font=("times new roman", "18", "bold"), bg='light grey')
                pred_l.place(x=600, y=300)
                
            col_1_l = Label(frame11, text=f'{name_x[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_1_l.place(x=350, y=150)
    
            col_2_l = Label(frame11, text=f'{name_x[1]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_2_l.place(x=350, y=210)
    
            col_3_l = Label(frame11, text=f'{name_y[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_3_l.place(x=350, y=300)
           
    
            col_1_en = StringVar()
            col_1_e = Entry(frame11, width=20, textvariable=col_1_en)
            col_1_e.place(x=600, y=155)
    
            col_2_en = StringVar()
            col_2_e = Entry(frame11, width=20, textvariable=col_2_en)
            col_2_e.place(x=600, y=210)

                
            result_btn = Button(frame11, text='Predict', bg='light grey', relief=GROOVE, command=pred_svm)
            result_btn.place(x=460, y=375)   
            
        def decision_fn():
            frame2.destroy()

            header_l = Label(win1, text='Decision Tree', font=("times new roman", "28", "bold"), bg='red', relief=RAISED, padx=6)
            header_l.pack(ipady=10)

            frame12 = Frame(win1, bg='white')
            frame12.pack(padx=190, pady=60, ipadx=50, ipady=50, expand=True, fill='both')

            def pred_decision():
                x_entry1 = col_1_e.get()
                x_entry2 = col_2_e.get()
         
               
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
                
                sc = StandardScaler()
                
                classifier = DecisionTreeClassifier(random_state=0)
                classifier.fit(x_train, y_train)
                
                y_pred = classifier.predict(x_test)
                
                
                pred_y = classifier.predict([[x_entry1, x_entry2]])
                print("Predicted Label for New Data:", pred_y)

            
                pred_l = Label(frame12, text=y_pred[0], font=("times new roman", "18", "bold"), bg='light grey')
                pred_l.place(x=600, y=250)
                
            col_1_l = Label(frame12, text=f'{name_x[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_1_l.place(x=350, y=150)
    
            col_2_l = Label(frame12, text=f'{name_x[1]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_2_l.place(x=350, y=200)
            
            col_3_l = Label(frame12, text=f'{name_y[0]}', font=("times new roman", "18", "bold"), bg='light grey')
            col_3_l.place(x=350, y=250)
    
            col_1_en = StringVar()
            col_1_e = Entry(frame12, width=20, textvariable=col_1_en)
            col_1_e.place(x=600, y=150)
    
            col_2_en = StringVar()
            col_2_e = Entry(frame12, width=20, textvariable=col_2_en)
            col_2_e.place(x=600, y=200)

                
            result_btn = Button(frame12, text='Predict', bg='light grey', relief=GROOVE, command=pred_decision)
            result_btn.place(x=460, y=375)
        # print('algo class',path,"X : ",x_col,"............Y : ",y_col)
        dataset = pd.read_csv(path)
        
        # print(dataset)
        # print()
        x = dataset.iloc[:, x_col].values
        y = dataset.iloc[:, y_col].values
       
        # print("x : ",x)
        # print("y : ",y)
        header_l = Label(frame2, text='PREDICTION', font=("times new roman", "28", "bold"), bg='light grey', relief=RAISED, padx=6)
        header_l.pack(ipady=20)

        frame3 = Frame(frame2, bg='white')
        frame3.pack(fill=BOTH, side=LEFT, expand=True, padx=10, pady=10)

        frame4 = Frame(frame2, bg='white')
        frame4.pack(fill=BOTH, side=LEFT, expand=True, padx=10, pady=10)

        regression_l = Label(frame3, text='Regression Model', font=("times new roman", "22", "bold"), bg='red', relief=RAISED,
                              padx=6)
        regression_l.pack(pady=10)

        lin_reg_btn = Button(frame3, text="Simple-Linear Regression", font=("times new roman", "14", "bold"), bg='light grey',
                              relief=GROOVE, command=slin_reg_fn,width=20)
        lin_reg_btn.place(x=170,y=100)

        mul_reg_btn = Button(frame3, text="Multi-Linear Regression", font=("times new roman", "14", "bold"), bg='light grey',
                              relief=GROOVE, command=mlin_reg_fn,width=20)
        mul_reg_btn.place(x=170,y=200)

        pol_reg_btn = Button(frame3, text="Polynomial Regression", font=("times new roman", "14", "bold"), bg='light grey',
                              relief=GROOVE, command=plin_reg_fn,width=20)
        pol_reg_btn.place(x=170,y=300)

        classification_l = Label(frame4, text='Classification Model', font=("times new roman", "22", "bold"), bg='red',
                                  relief=RAISED, padx=6)
        classification_l.pack(pady=10)

        log_cls = Button(frame4, text="Logistic Regression", font=("times new roman", "14", "bold"), bg='light grey',
                          relief=GROOVE, command=lcls_fn,width=20)
        log_cls.place(x=170,y=100)

        nav_bs_btn = Button(frame4, text="Naive - Bayes", font=("times new roman", "14", "bold"), bg='light grey', relief=GROOVE,
                            command=naive_fn,width=20)
        nav_bs_btn.place(x=170,y=200)
        
        knn_bs_btn = Button(frame4, text="KNN Algorithm", font=("times new roman", "14", "bold"), bg='light grey', relief=GROOVE,width=20,command=knn_fn)
        knn_bs_btn.place(x=170,y=300)
        
        svc_bs_btn = Button(frame4, text="svm", font=("times new roman", "14", "bold"), bg='light grey', relief=GROOVE,width=20,command=svm_fn)
        svc_bs_btn.place(x=170,y=400)
        
        svc_bs_btn = Button(frame4, text="decision tree", font=("times new roman", "14", "bold"), bg='light grey', relief=GROOVE,width=20,command=decision_fn)
        svc_bs_btn.place(x=170,y=500)


        win1.mainloop()


obb = base()
obb.base_fn()
