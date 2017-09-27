import pyflux as pf

import pandas as pd
import matplotlib as mlp
mlp.use('Agg')
from matplotlib import pyplot as plt

import time

import numpy as np
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')

# for db connection
import pymysql
from sqlalchemy import create_engine
db_conx = create_engine('mysql+pymysql://root:abhinav@7@localhost:3306/prediction_output', echo=False)

holiday_list_file = 'us_holidays.csv'


def build_data_frame(data_frame):

    # convert the string in the format of US locale format like 1,029 strings into 1029
    sales_data = [x if type(x) != str else locale.atoi(x) for x in data_frame.sales]
    # print sales_data
    budget_data = [x if type(x) != str else locale.atoi(x) for x in data_frame.budget]
    data_frame.budget = pd.DataFrame(budget_data).values
    data_frame.sales = pd.DataFrame(sales_data).values
    # print data_frame.sales
    # make all of the data as float64
    data_frame[['sales', 'promo', 'budget', 'retail']] = data_frame[
        ['sales', 'promo', 'budget', 'retail']
    ].apply(
        pd.to_numeric
    )
    if 'week' in data_frame.keys():
        try:
            data_frame.week = pd.to_datetime(data_frame.week+'-1', format='%yw%W-%w')
        except:
            data_frame.week = pd.to_datetime(data_frame.week + '-1', format='w%W %Y-%w')
        # data_frame['week'] = pd.DatetimeIndex(data_frame['week'])
        data_frame = data_frame.set_index(data_frame['week'])

        # to fill the NaN values with the mean value of that column
    data_frame = data_frame.fillna(value=data_frame.mean())
    # print data_frame
    return data_frame


def holiday_mapping(holiday_list_file):
    holiday_data_frame = pd.read_csv(holiday_list_file)
    holiday_data_frame.date = pd.to_datetime(holiday_data_frame.date, format="%Y-%m-%d")
    list_of_unique_holidays = holiday_data_frame.holiday.unique()
    holiday_dict = {}
    holiday_id = []
    for index, i in enumerate(list_of_unique_holidays):
        holiday_dict[i] = index + 1
    for i in holiday_data_frame.holiday:
        if i in holiday_dict.keys():
            holiday_id.append(holiday_dict[i])
    # print holiday_id
    holiday_data_frame['holiday'] = np.array(holiday_id)
    return holiday_data_frame


def separate_multiple_products(filename):
    data_frame_dict = {}
    data_frame = pd.read_csv(filename)
    data_frame.columns = map(str.lower, data_frame.columns)

    # make all of the entries in the variables column as lower case
    data_frame.variables = data_frame['variables'].str.lower()

    # holiday_data_frame = holiday_mapping(holiday_list_file)

    data_frame2 = data_frame.drop(['cid', 'region', 'pid', 'lid', 'variables'], axis=1)

    count = 0
    # last_week = data_frame.keys()[-1]
    # last_date = datetime.datetime.strptime(last_week + '-0', '%yw%W-%w').date()
    # first_date = datetime.datetime.strptime(data_frame2.keys()[0] + '-0', '%yw%W-%w').date()

    for i, j, k, l, m in zip(data_frame.pid, data_frame.region, data_frame.lid, data_frame.cid, data_frame.variables):
        key = ','.join((map(str, [i, j, k, l])))
        if key in data_frame_dict.keys():
            data_frame_dict[key][m] = list(data_frame2.loc[count].values)
        else:
            data_frame_dict[key] = {m: list(data_frame2.loc[count].values)}
        count += 1
    # print data_frame_dict
    weeks = data_frame2.keys().values
    # print type(weeks)
    # data_frame_dict['week'] = [data_frame2.keys().values]
    for key in data_frame_dict.keys():
        sales_data = data_frame_dict[key]
        try:
            new_data_frame = pd.DataFrame({'promo': sales_data['promo'], 'retail': sales_data['retail'], 'budget': sales_data['budget'], 'sales': sales_data['sales'],
                                           'week': weeks})
        except:
            new_data_frame = pd.DataFrame(
                {'promo': sales_data['promo'], 'retail': sales_data['retail price'], 'budget': sales_data['budget'],
                 'sales': sales_data['sales'],
                 'week': weeks})
        # new_data_frame.add_prefix()
        # print new_data_frame
        yield new_data_frame, key.split(','), data_frame.keys()[:4].values


def arima(filename, parameters, weeks):
    output_frames = []
    predictions_frames = []
    data_frames = []
    for frame, labels_values, labels in separate_multiple_products(filename):
        data_frame = build_data_frame(frame)
        data_frames.append(data_frame)
        # print data_frame
        integ, ar, ma = map(int, [i.strip() for i in parameters.split(',')])
        arima = pf.ARIMA(data_frame, integ=integ, ar=ar, ma=ma, family=pf.Normal(), target='sales')
        # arima = pf.ARIMA(data_frame, integ=0, ar=5, ma=0, family=pf.Normal(), target='sales')
        x = arima.fit()
        x.summary()
        print (x.loglik, x.max_lag)

        # arima.plot_fit()

        predictions_frame = arima.predict(weeks)
        # print arima.predict_is(h=2)
        predictions_frames.append(predictions_frame)
        # arima.plot_z(figsize=(10,6))
        # arima.plot_predict_is(3)
        # arima.plot_predict(5, past_values=0, intervals=False)
        # print (type(predictions_frame))
        # file_path = '/home/trusty/PycharmProjects/sales_forecast/Assist/codes/output.csv'
        output_frames.append(build_output_frame(predictions_frame, labels, labels_values))

    final_data_frame = pd.concat(output_frames)
    final_data_frame.to_csv('output.csv')

    time_stamp = time.strftime("%d/%m/%Y").__str__().replace('/', '') + \
                 time.strftime("%H:%M:%S").__str__().replace(':', '')

    final_data_frame.to_sql(name=time_stamp, con=db_conx, if_exists='replace', index=False)

    graph_data = []
    for i, j in zip(data_frames, predictions_frames):
        graph_data.append(i.sales.tolist() + j.sales.tolist())
    # print graph_data
    legend_labels = []
    for i in output_frames:
        legend_labels.append('pid:{0},region:{1},lid:{2},cid:{3}'.format(i.pid[0], i.region[0], i.lid[0], i.cid[0]))
    # print legend_labels
    xtick_labels = [i.date() for i in data_frame.index.tolist()] + [i.date() for i in predictions_frame.index.tolist()]
    # print x_tick_labels
    plot_graph(
        graph_data,
        title='Sales Forecasting',
        x_label='Weeks',
        y_label='Sales Quantity',
        legend_labels=legend_labels,
        xtick_labels=xtick_labels,
        filename='sales.png'
    )


def build_output_frame(predictions_frame, labels, labels_values):
    output_frame = pd.DataFrame([
        [int(labels_values[0]), int(labels_values[1]), int(labels_values[2]), int(labels_values[3])]
    ],
        columns=[labels[0], labels[1], labels[2], labels[3]])
    output_frame = output_frame.join(
        pd.DataFrame([predictions_frame.sales.values], columns=predictions_frame.index.date),
        how='right')
    return output_frame


def plot_graph(data, title='Graph', x_label='X-axis', y_label='Y-axis', xtick_labels=[], legend_labels=[], filename='fig.png'):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80, squeeze=True)
    x_axis_len = len(data[0])
    total_plots = len(data)
    if total_plots == 1:
        # graph_1 = ax.bar(np.arange(x_axis_len), height=data[0], width=0.35, color='b')
        # autolabel(graph_1, ax)
        ax.plot(data[0])
    elif total_plots > 1:
        for i in range(total_plots):
            ax.plot(data[i], label=legend_labels[i])

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    step = len(data[0])/22
    if step >= 10:
        pass
    else:
        step = 1
    ax.set_xticks(np.arange(x_axis_len)[::step])
    ax.set_xticklabels(xtick_labels[::step])
    # Tell matplotlib to interpret the x-axis values as dates
    ax.xaxis_date()

    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()
    ax.legend()
    # plt.show()
    plt.savefig(filename, bbox_inches='tight')


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.005*height,
                '%d' % int(height),
                ha='center', va='bottom')

arima('/home/trusty/PycharmProjects/sales_forecast/Assist/codes/ML_NEW_IP.csv', '0,2,0', 5)