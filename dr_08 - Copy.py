import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
import numpy as np
from math import ceil
matplotlib.style.use('ggplot')
import zipfile, requests, io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn import svm


def main():
    #build_dataset()
    predict_price()


# Perform linear regression on LMP time series
def predict_price():
    window = [5, 24*4, 5]   # Number of data points (hourly) to use in the X vector for each model:
                            # [trend, frequency, temperature]
    forecast = 24*3         # length of forecast
    m = 24*14               # train the model using m previous examples
    offset = 24*(14)        # move starting point in the time series by n hours
    days_missing = 24*4
    coefs = 8               # number of harmonics we want to include (max. 11)

    [x,y,t,time_series,siteID,plot_begin,plot_end] = build_vectors(window)

    # **********************************************************************8
    ### 1. Simple Model

    ### 1.1 Polynomial model ---------------

    # Split data set into vectors used for training and prediction
    t_plot = t[max(window):-days_missing]
    x_train = x[offset:m + offset]
    y_train = y[offset:m + offset]
    x_test = x[m + offset:m + offset + forecast]
    y_real = y[m + offset:m + offset + forecast]

    # Train model, predict, and plot results
    plot_info = 211
    plot_title = 'Simple Model'
    fig = plt.figure(figsize=(8, 6))
    order = 2
    pr1 = make_pipeline(PolynomialFeatures(order), LinearRegression())
    [y_test, mae] = train_model(pr1,x_train,y_train,x_test,y_real)
    plot_data(plot_title,fig,plot_info,y, y_test, y_train, t_plot, offset, m, forecast, t, mae, siteID, plot_begin, plot_end)

    ### 1.2 SVM -----------------
    plot_info = 212
    svr1 = svm.SVR(kernel='linear', C=1e2, gamma=1e-3)
    [y_test, mae] = train_model(svr1, x_train, y_train, x_test, y_real)
    plot_data(plot_title,fig,plot_info, y, y_test, y_train, t_plot, offset, m, forecast, t, mae, siteID, plot_begin, plot_end)

    # **********************************************************************8
    ### 2. Fourier Analysis

    ### 2.1 Linear regression model ---------------
    x2 = []
    # Append frequency domain data to the existing X array
    for i in range(max(window), time_series.size):
        x_new = ts_fft(time_series[i - window[1]:i], coefs)
        x2.append(np.hstack((x[i - max(window)], x_new.T)))
    x2 = np.asarray(x2)
    x_train = x2[offset:m + offset]
    x_test = x2[m + offset:m + offset + forecast]

    plot_info = 211
    plot_title = 'Fourier Analysis'
    fig = plt.figure(figsize=(8, 6))
    lr2 = LinearRegression(normalize=True)
    [y_test, mae] = train_model(lr2, x_train, y_train, x_test, y_real)
    plot_data(plot_title,fig, plot_info, y, y_test, y_train, t_plot, offset, m, forecast, t, mae, siteID, plot_begin, plot_end)

    # 2.2 SVM ------------------------------------
    svr2 = svm.SVR(kernel='rbf', C=1e3, gamma=1e-5)
    plot_info = 212
    [y_test, mae] = train_model(svr2, x_train, y_train, x_test, y_real)
    plot_data(plot_title,fig, plot_info, y, y_test, y_train, t_plot, offset, m, forecast, t, mae, siteID, plot_begin, plot_end)


    # **********************************************************************
    ### 3. Temperature Data

    ### Get temperature data
    # --------------------------------------------------------------------------
    temp = get_temp()
    temp = temp[7:]   # 7 hours shift for PST. needs to line up with $/MWh time series
    temp = np.concatenate((temp[:29 * 24], temp[31 * 24:58 * 24], temp[59 * 24:89 * 24 + (window[2] + 1)]))
    # temp = np.concatenate((temp[:29 * 24], temp[31 * 24:58 * 24], temp[59 * 24:89 * 24 + (2*window[2] + 1)]))

    ### 3.1 Linear regression model ---------------
    x3 = []
    # Append temperature data to X array
    for i in range(max(window), len(x2)):
    # for i in range(max(window),time_series.size):
        temp_window = temp[i - window[2]: i + (window[2] + 1)]
        # temp_window = temp[i : i + (2*window[2] + 1)]
        x3.append(np.hstack((x2[i - max(window)], temp_window)))
    x3 = np.asarray(x3)
    x_train = x3[offset:m + offset]
    x_test = x3[m + offset:m + offset + forecast]

    plot_info = 211
    plot_title = 'Temperature Data'
    fig = plt.figure(figsize=(8, 6))
    lr3 = LinearRegression(normalize=True)
    [y_test, mae] = train_model(lr3, x_train, y_train, x_test, y_real)
    plot_data(plot_title,fig, plot_info, y, y_test, y_train, t_plot, offset, m, forecast, t, mae, siteID, plot_begin, plot_end)

    # 3.2 SVM ------------------------------------
    svr3 = svm.SVR(kernel='rbf', C=1e3, gamma=1e-5)
    plot_info = 212
    [y_test, mae] = train_model(svr3, x_train, y_train, x_test, y_real)
    plot_data(plot_title,fig, plot_info, y, y_test, y_train, t_plot, offset, m, forecast, t, mae, siteID, plot_begin, plot_end)

    # Calculate error for each model
    find_error(x, y, m, x2, x3, pr1, lr2, lr3)

    plt.show()


def find_error(x,y,m,x2,x3,pr1,lr2,lr3):
    # Find overall error for each model for all predictions across the time series
    errors = [[], [], []] # [ [simple], [frequency], [temperature] ]
    # for i in range(len(y)-m-days_missing):
    for i in range(0):
        x_train = x[i: m + i]
        y_train = y[i: m + i]
        x_test = x[m + i]
        y_real = y[m + i]
        pr1.fit(x_train, y_train)
        y_simple = pr1.predict(x_test)
        errors[0].append(abs(y_real - y_simple))
        # errors[0].append(np.round((abs(y_simple - y_real) / (y_real)) * 100, 2))

        x_train = x2[i: m + i]
        x_test = x2[m + i]
        lr2.fit(x_train, y_train)
        y_simple = lr2.predict(x_test)
        errors[1].append(abs(y_real - y_simple))
        # errors[1].append(np.round((abs(y_simple - y_real) / (y_real)) * 100, 2))

        x_train = x3[i: m + i]
        x_test = x3[m + i]
        lr3.fit(x_train, y_train)
        y_simple = lr3.predict(x_test)
        errors[2].append(abs(y_real - y_simple))
        # errors[2].append(np.round((abs(y_simple - y_real) / (y_real)) * 100, 2))

    # a = np.asarray(errors)
    # np.savetxt("error_lr_abs_abs.csv", a, delimiter=",")

    errors = np.loadtxt(open("error_lr_abs_abs.csv", "rb"), delimiter=",")

    errors = np.asarray(errors)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(errors[0, :], '--', color='#006b12', label='Simple Model')  # green
    plt.plot(errors[1, :], '--', color='#0000ff', label='Fourier Analysis')  # blue
    plt.plot(errors[2, :], '--', color='#cc0000', label='Temperature Data')  # red
    plt.title('Error Percentage for each DR Prediction Model')
    plt.legend(loc='upper right')

    print('Average Error for Simple Model: ', np.mean(errors[0, :]))
    print('Average Error with Fourier Analysis: ', np.mean(errors[1, :]))
    print('Average Error with Temperature Data: ', np.mean(errors[2, :]))


def train_model(mod,x_train,y_train,x_test,y_real):
    # Train model and predict new values
    mod.fit(x_train, y_train)
    y_test = mod.predict(x_test)
    # Get Mean Absolute Error
    mae = 0
    if len(y_test) > 1:
        mae = str(round(mean_absolute_error(y_real, y_test), 2))
    if len(y_test) == 1:  # sk-learn's mae does not handle vectors with a single value
        mae = abs(y_real - y_test[0])
    return y_test, mae


def plot_data(plot_title,fig,plot_info,y,y_test,y_train,t_plot,offset,m,forecast,t,mae,siteID,plot_begin,plot_end):
    # Plots the $/MWh with training data and forecast overlaid on top of the ground truth
    t_train = t_plot[offset:m + offset]
    t_test = t_plot[m + offset:m + offset + forecast]
    ax = fig.add_subplot(plot_info)
    plt.plot(t_plot, y, '-o', label='True demand', color='#377EB8', linewidth=2)
    plt.plot(t_test, y_test, '-o', color='#EB3737', linewidth=3, label='Prediction')
    plt.plot(t_train, y_train, label='Train data', color='#3700B8', linewidth=2)
    plt.legend(loc='lower right')
    # Plot Linear Regression and SVM performance side by side as subplots
    if plot_info == 211:
        plt.title('Linear Regression with ' + plot_title + ',           MAE: ' + mae)
    if plot_info == 212:
        plt.title('SVM with ' + plot_title + ',           MAE: ' + mae)
        plt.suptitle(siteID + '     ' + str(plot_begin.date()) + ' - ' + str(plot_end.date()), fontsize=10)
        ax.set_ylabel('$ / MWh')
        fig.autofmt_xdate()
        plt.subplots_adjust(top=0.88)
        plt.subplots_adjust(bottom=0.12)


def build_vectors(window):
    # Creates the X and Y vectors from OASIS data stored locally as a csv file
    file_dir = 'C:/Users/laguilar/PycharmProjects/Demand_Response/'
    file = 'BAYSHOR2_1_N001 -- 20140101-20140401.csv'
    name_len = 15
    siteID = file[:name_len]
    # Get length of time series from file name
    plot_begin = dt.datetime.strptime(file[name_len + 4:name_len + 12], '%Y%m%d')
    plot_end = dt.datetime.strptime(file[name_len + 13:name_len + 21], '%Y%m%d')
    print(siteID, plot_begin, plot_end)
    t = plot_end - plot_begin
    t = [plot_begin + dt.timedelta(hours=i) for i in range((t.days) * 24)]
    data = np.loadtxt(file_dir + file, delimiter=',')
    # Load time series and create lagged observations and target predictions
    time_series = data.flatten()
    x = []
    y = []
    for i in range(max(window), time_series.size):
        x.append(time_series[i - window[0]:i])
        y.append(time_series[i])
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y, t, time_series, siteID, plot_begin, plot_end
    # return {
    #     'x': x,
    #     'y': y,
    # }


def ts_fft(ts,coefs):
    ### Get FFT
    ftrain = np.fft.fft(ts)   # ts should be a multiple a 24, giving 11 harmonics: (24/2)-1
    # Get rid of imaginary part (right hand plane)
    f_dummy = ftrain[:int(len(ftrain)/2)]
    harmonics = []
    cycle = int(len(ts) / 24)
    # Find harmonics (spaced out evenly at same interval)
    for i in range(1,coefs+1):
        harmonics.append(f_dummy[i*cycle])
    return np.asarray(harmonics)


def get_temp():
    # Load csv file and extract temperature data as a numpy array
    file = 'weather_rwc_01-04'
    file_dir = 'C:/Users/laguilar/Downloads/'
    file = file + '.csv'
    df = pd.read_csv(file_dir + file)
    df = df[df.Type == 'FM-13']
    df = df['Temp']
    return df.as_matrix()


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


def build_dataset():
    print('Main ---')
    srt_date = '20140501'
    end_date = '20140531'
    days = (dt.datetime.strptime(end_date,"%Y%m%d") - dt.datetime.strptime(srt_date,"%Y%m%d")).days
    print(days)
    sites = ['REDWOOD_6_N001']
    from_file = 1
    if from_file == 0:
        print('Batch: ', 1)
        td = 10
        site_df = build_df(srt_date, td, sites, from_file)
        ds = make_dataset(site_df, td)
        print('iter: ',ceil(days/td) )
        for i in range( int(days/td)-1 ):
            print('Batch: ',i+2)
            srt_date = dt.datetime.strptime(srt_date,"%Y%m%d") + dt.timedelta(days=td)
            srt_date = str(srt_date.year) + str(srt_date.month) + str(srt_date.day)
            site_df = build_df(srt_date, td, sites, from_file)
            ds = np.row_stack((ds,make_dataset(site_df, td)) )
    else:
        site_df = build_df(srt_date, days, sites, from_file)
        #days = 30
        ds = make_dataset(site_df,days)
    name = sites[0] + ' -- ' + srt_date + '-' + end_date
    print(name,ds.shape)
    np.savetxt(name, ds, delimiter=",")


def build_df(start_date, days, sites, from_file):

    if from_file:
        file = '20160702_20160801_PRC_LMP_DAM_20170531_16_43_27_v1'
        file_dir = 'C:/Users/laguilar/Downloads/'+file+'/'
        file = file+'.csv'
        df = pd.read_csv(file_dir + file)

        start_date = file[0:8]
        end_date = file[9:17]
        days = (dt.datetime.strptime(end_date,'%Y%m%d') - dt.datetime.strptime(start_date,'%Y%m%d')).days

    else:
        dfs = get_data(start_date, days,sites)
        df = dfs[0]

    print('actual days: ',days)
    site_label = df['NODE'].unique()
    print('num sites: ',len(site_label))
    site_df = []
    price_label = df['XML_DATA_ITEM'].unique()
    for site in site_label:
        site_cat = df.loc[df['NODE'] == site]
        price_df = []
        for label in price_label:
            price_cat = site_cat.loc[site_cat['XML_DATA_ITEM'] == label]
            day_df = []
            for i in range(days):
                find_date = dt.datetime.strptime(start_date,"%Y%m%d") + dt.timedelta(days=i)
                find_date = str(find_date.date())
                date_cat = price_cat.loc[price_cat['OPR_DT'] == find_date]
                date_cat = date_cat.sort_values(by='OPR_HR')
                day_df.append(date_cat)
            price_df.append(day_df)
        site_df.append(price_df)

    ### Structure: site_df[ price_df[ day_df[ data sorted by 'OPR_HR' ] ] ]
    #plot_things(site_df,site_label,days)
    return site_df


def make_dataset(site_df,days):
    # Will receive one site only, and we're only interested in the total price
    # The remaining list is the sorted days
    print(len(site_df[0][0]))
    dataset = np.array(site_df[0][0][1]['MW'])
    print('nr', dataset.shape)

    for i in range(2,days):
        new_row = np.array(site_df[0][0][i]['MW'])
        # pad array in case there are fewer than 24 entries
        if new_row.size<24:
            new_row = np.concatenate((new_row,np.zeros(24-new_row.size)))
        dataset = np.row_stack((dataset,new_row[:24]))
    return dataset


def get_data(start_date, days, sites):
    query = 'PRC_LMP'
    tz = 'T07:00-0000'
    td = dt.timedelta(days=days)
    end_time = dt.datetime.strptime(start_date,"%Y%m%d") + td
    print(end_time)
    end_time = str(end_time.date())
    end_time = end_time[0:4] + end_time[5:7] + end_time[8:10]
    print(end_time)
    if sites[0] == '':
        get_sites = 'grp_type=ALL_APNODES'
    else:
        get_sites = 'node='+sites[0]

    print('for url: '+start_date+'     '+end_time)
    url = 'http://oasis.caiso.com/oasisapi/SingleZip?queryname='+query+\
          '&startdatetime='+start_date+tz+'&enddatetime='+end_time+tz+'&version=1&market_run_id=DAM&'+\
          get_sites+'&resultformat=6'
    r = requests.get(url)
    r1 = io.BytesIO(r.content)
    print(url)
    print(r1)
    zip = zipfile.ZipFile(r1)
    zip_names = zip.namelist()
    print(zip_names)
    df = []
    for file in zip_names:
        text = zip.read(file)
        text_csv = io.StringIO(text.decode(encoding='UTF-8'))
        df.append(pd.read_csv(text_csv))
    return df


def plot_things(site_df,site_label,days):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    small = 0
    small_size = 3
    if small:
        sites = small_size
    else:
        sites = len(site_label)
    # number of sites to plot
    # NOTE: not interested in other price_df components; index can remain price_df[0]
    for i in range(sites):
        for day in range(days):
            plt.plot(site_df[i][0][0]['OPR_HR'], site_df[i][0][day]['MW'], '-o',
                     label=site_label[i] + '   Day : ' + str(day))
    plt.legend()
    ax.set_title('LMP')
    ax.set_ylabel('$ / MWh')
    ax.set_xlabel('Time (24h)')
    #plt.show()


# ---------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()